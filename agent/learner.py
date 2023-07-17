

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch.multiprocessing as mp
import torch.distributed.rpc as rpc
from torch.distributed.rpc import RRef
from torch.distributed.rpc.functions import async_execution
from torch.futures import Future

import gym
import numpy as np
import threading
import time
import random
from copy import deepcopy

from .actor import Actor
from .replaybuffer import ReplayBuffer

from models import Model
from curiosity import EpisodicNovelty, LifelongNovelty
from utils import compute_retrace_loss, value_rescaling, inverse_value_rescaling


class Learner:
    """
    Main class used to train the agent. Called by rpc remote.
    Call run() to start the main training loop.

    Parameters:
        env_name (string): Environment name in gym[atari]
        size (int): The size of the buffer in ReplayBuffer
        B (int): Batch size for training
        burnin (int): Length of burnin, concept from R2D2 paper
        rollout (int): Length of rollout, concept from R2D2 paper

    """
    epsilon = 1
    epsilon_min = 0.1
    epsilon_decay = 0.0001

    lr = 1e-4
    discount = 0.95
    beta = 0.05  # 0.2

    update_every = 400
    save_every = 100
    device = "cuda"

    def __init__(self,
                 env_name,
                 size,
                 B,
                 burnin,
                 rollout
                 ):
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        self.size = size
        self.B = B

        # models
        self.action_size = gym.make(env_name).action_space.n
        model = Model(action_size=self.action_size)

        # episodic novelty module / lifelong novelty module
        self.episodic_novelty = EpisodicNovelty(action_size=self.action_size)
        self.lifelong_novelty = LifelongNovelty()

        self.model = nn.DataParallel(deepcopy(model)).cuda()
        self.target_model = nn.DataParallel(deepcopy(model)).cuda()
        self.eval_model = nn.DataParallel(deepcopy(model)).cuda()

        # set model modes
        self.model.train()
        self.target_model.eval()
        self.eval_model.eval()

        # locks
        self.lock = mp.Lock()
        self.lock_model = mp.Lock()

        # hyper-parameters
        self.burnin = burnin
        self.rollout = rollout
        self.T = burnin + rollout

        # optimizer and loss functions
        self.opt = optim.Adam(self.model.parameters(), lr=self.lr)

        # queues
        self.sample_queue = mp.Queue()
        self.batch_queue = mp.Queue()
        self.priority_queue = mp.Queue()

        self.batch_queue = mp.Queue(8)
        self.priority_queue = mp.Queue(8)

        # params, batched_data (feeds batch), pending_rpcs (answer calls)
        self.batch_data = []

        # start replay buffer
        self.replay_buffer = ReplayBuffer(size=size,
                                          B=B,
                                          T=burnin+rollout,
                                          discount=self.discount,
                                          sample_queue=self.sample_queue,
                                          batch_queue=self.batch_queue,
                                          priority_queue=self.priority_queue
                                          )

        # start actors
        self.future1 = Future()
        self.future2 = Future()

        self.pending_rpc = None
        self.await_rpc = False

        self.actor_rref = self.spawn_actor(learner_rref=RRef(self),
                                           env_name=env_name
                                           )

        self.updates = 0

    @staticmethod
    def spawn_actor(learner_rref, env_name):
        """
        Start actor by calling actor.remote().run()
        Actors communicate with learner through rpc and RRef

        Parameters:
        learner_rref (RRef): learner RRef for actor to reference the learner
        tickers (List[2]): A list of tickers e.g. ["AAPL", "GOOGL"]
        d_model (int): Dimension of model
        state_len (int): Length of recurrent state

        Returns:
        actor_rref (RRef): to reference the actor from the learner
        """
        actor_rref = rpc.remote("actor",
                                Actor,
                                args=(learner_rref,
                                      0,
                                      env_name
                                      ),
                                timeout=0
                                )
        actor_rref.remote().run()

        return actor_rref

    @async_execution
    def queue_request(self, *args):
        """
        Called by actor asynchronously to queue requests

        Returns:
        future (Future.wait): Halts until value is ready
        """
        future = self.future1.then(lambda f: f.wait())
        with self.lock:
            self.pending_rpc = args

        return future

    @async_execution
    def return_episode(self, episode):
        """
        Called by actor to asynchronously to return completed Episode
        to Learner

        Returns:
        future (Future.wait): Halts until value is ready
        """
        future = self.future2.then(lambda f: f.wait())
        self.sample_queue.put(episode)
        self.await_rpc = True

        # Never Give Up
        with self.lock_model:
            self.episodic_novelty.reset()

        return future

    @torch.inference_mode()
    def get_policy(self, obs, state1, state2):
        self.epsilon -= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)

        with self.lock_model:
            qe, qi, state1, state2 = self.eval_model(obs, state1, state2)
            q_values = value_rescaling(inverse_value_rescaling(qe) + self.beta * inverse_value_rescaling(qi))
            # q_values = qe + self.beta * qi

            intr_e = self.episodic_novelty.get_reward(obs)
            intr_l = self.lifelong_novelty.get_reward(obs)
            intr = intr_e * intr_l

            state1 = (state1[0].detach().cpu().numpy(),
                      state1[1].detach().cpu().numpy())
            state2 = (state2[0].detach().cpu().numpy(),
                      state2[1].detach().cpu().numpy())

        if random.random() <= self.epsilon:
            action = random.randrange(0, self.action_size)
            prob = self.epsilon / self.action_size

            return action, prob, state1, state2, intr

        # get action and probability of that action according to Agent57 (pg 19)
        action = torch.argmax(q_values.squeeze()).detach().cpu().squeeze().item()
        prob = 1 - (self.epsilon * ((self.action_size - 1) / self.action_size))

        return action, prob, state1, state2, intr

    def get_action(self, obs, state1, state2):
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0) / 255.
        state1 = (torch.tensor(state1[0], dtype=torch.float32, device=self.device),
                  torch.tensor(state1[1], dtype=torch.float32, device=self.device))
        state2 = (torch.tensor(state2[0], dtype=torch.float32, device=self.device),
                  torch.tensor(state2[1], dtype=torch.float32, device=self.device))

        return self.get_policy(obs, state1, state2)

    def answer_requests(self):
        """
        Thread to answer actor requests from queue_request and return_episode.
        Loops through with a time gap of 0.0001 sec
        """

        while True:
            time.sleep(0.0001)

            with self.lock:

                # clear self.future2 (store episodes)
                if self.await_rpc:
                    self.await_rpc = False

                    future = self.future2
                    self.future2 = Future()
                    future.set_result(None)

                # clear self.future1 (answer requests)
                if self.pending_rpc is not None:
                    results = self.get_action(*self.pending_rpc)
                    self.pending_rpc = None

                    future = self.future1
                    self.future1 = Future()
                    future.set_result(results)

    def prepare_data(self):
        """
        Thread to prepare batch for update, batch_queue is filled by ReplayBuffer
        Loops through with a time gap of 0.01 sec
        """

        while True:
            time.sleep(0.001)

            if not self.batch_queue.empty() and len(self.batch_data) < 4:
                data = self.batch_queue.get_nowait()
                self.batch_data.append(data)

    def run(self):
        """
        Main training loop. Start ReplayBuffer threads, answer_requests thread,
        and prepare_data thread. Then starts training
        """
        self.replay_buffer.start_threads()

        inference_thread = threading.Thread(target=self.answer_requests, daemon=True)
        inference_thread.start()

        background_thread = threading.Thread(target=self.prepare_data, daemon=True)
        background_thread.start()

        while True:

            while not self.batch_data:
                time.sleep(0.001)
            block = self.batch_data.pop(0)

            self.update(obs=block.obs,
                        actions=block.actions,
                        probs=block.probs,
                        extr=block.extr,
                        intr=block.intr,
                        states1=block.states1,
                        states2=block.states2,
                        dones=block.dones,
                        idxs=block.idxs
                        )

    def update(self, obs, actions, probs, extr, intr, states1, states2, dones, idxs):
        """
        An update step. Performs a training step, update new recurrent states,
        soft update target model and transfer weights to eval model
        """
        loss, new_states1, new_states2 = self.train_step(
            obs=obs.cuda(),
            actions=actions.cuda(),
            probs=probs.cuda(),
            extr=extr.cuda(),
            intr=intr.cuda(),
            states1=(states1[0].cuda(), states1[1].cuda()),
            states2=(states2[0].cuda(), states2[1].cuda()),
            dones=dones.cuda()
        )
        intr_loss = self.train_novelty_step(
            obs=obs.cuda(),
            actions=actions.cuda()
        )

        # reformat List[Tuple(Tensor, Tensor)] to array of shape (bsz, block_len+n_step, 2, dim)
        states11, states12 = zip(*new_states1)
        states11 = torch.stack(states11).transpose(0, 1).cpu().numpy()
        states12 = torch.stack(states12).transpose(0, 1).cpu().numpy()
        new_states1 = np.stack([states11, states12], 2)

        states21, states22 = zip(*new_states2)
        states21 = torch.stack(states21).transpose(0, 1).cpu().numpy()
        states22 = torch.stack(states22).transpose(0, 1).cpu().numpy()
        new_states2 = np.stack([states21, states22], 2)

        # update new states to buffer
        self.priority_queue.put((idxs, new_states1, new_states2, loss, intr_loss, self.epsilon))

        # soft update target model
        if self.updates % self.update_every == 0:
            self.hard_update(self.target_model, self.model)

        # transfer weights to eval model
        with self.lock_model:
            self.hard_update(self.eval_model, self.model)
            self.episodic_novelty.update_eval()
            self.lifelong_novelty.update_eval()

        return loss, intr_loss

    def train_step(self, obs, actions, probs, extr, intr, states1, states2, dones):
        """
        Accumulate gradients to increase batch size
        Gradients are cached for n_accumulate steps before optimizer.step()

        Args:
            obs (block+1, B, channels, h, w]): tokens
            actions (block+1, B): actions
            probs (block+1, B): probs
            extr (block, B): extrinsic rewards
            intr (block, B): extrinsic rewards
            states1 (B, dim): recurrent states
            states2 (B, dim): recurrent states
            dones (block+1, B): boolean indicating episode termination

        Returns:
            loss (float): Loss of critic model
            bert_loss (float): Loss of bert masked language modeling
            new_states1 (B, dim): for lstm
            new_states2 (B, dim): for lstm
        """

        with torch.no_grad():
            state1 = (states1[0].detach().clone(), states1[1].detach().clone())
            state2 = (states2[0].detach().clone(), states2[1].detach().clone())

            new_states1, new_states2 = [], []
            for t in range(self.burnin):
                new_states1.append((state1[0].detach(), state1[1].detach()))
                new_states2.append((state2[0].detach(), state2[1].detach()))

                _, _, state1, state2 = self.target_model(obs[t], state1, state2)

            target_q1, target_q2 = [], []
            for t in range(self.burnin, self.T+1):
                new_states1.append((state1[0].detach(), state1[1].detach()))
                new_states2.append((state2[0].detach(), state2[1].detach()))

                target_q1_, target_q2_, state1, state2 = self.target_model(obs[t], state1, state2)
                target_q1.append(target_q1_)
                target_q2.append(target_q2_)

            target_q1 = torch.stack(target_q1)
            target_q2 = torch.stack(target_q2)

        self.model.zero_grad()

        state1 = (states1[0].detach().clone(), states1[1].detach().clone())
        state2 = (states2[0].detach().clone(), states2[1].detach().clone())

        for t in range(self.burnin):
            _, _, state1, state2 = self.model(obs[t], state1, state2)

        q1, q2 = [], []
        for t in range(self.burnin, self.T+1):
            q1_, q2_, state1, state2 = self.model(obs[t], state1, state2)
            q1.append(q1_)
            q2.append(q2_)

        q1 = torch.stack(q1)
        q2 = torch.stack(q2)

        pi_t1 = F.softmax(q1, dim=-1)
        pi_t2 = F.softmax(q2, dim=-1)

        discount_t = (~dones).float() * self.discount

        extr_loss = compute_retrace_loss(
            q_t=q1[:-1],
            q_t1=target_q1[1:],
            a_t=actions[:-1],
            a_t1=actions[1:],
            r_t=extr,
            pi_t1=pi_t1[1:],
            mu_t1=probs[1:],
            discount_t=discount_t
        )
        intr_loss = compute_retrace_loss(
            q_t=q2[:-1],
            q_t1=target_q2[1:],
            a_t=actions[:-1],
            a_t1=actions[1:],
            r_t=intr,
            pi_t1=pi_t2[1:],
            mu_t1=probs[1:],
            discount_t=discount_t
        )

        loss = extr_loss + intr_loss

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        loss = loss.item()
        return loss, new_states1, new_states2

    def train_novelty_step(self, obs, actions):
        emb_loss = self.train_emb_step(obs, actions)
        lifelong_loss = self.train_lifelong_step(obs)

        return emb_loss + lifelong_loss

    def train_emb_step(self, obs, actions):
        """
        Args:
            obs (torch.Tensor): shape (block+1, bsz, 4, 105, 80)
            actions (torch.Tensor): shape (block+1, bsz, 1)
        """

        # Get obs, next_obs and action, and flatten time and batch dimension
        obs1 = torch.flatten(obs[:-1, ...], 0, 1)
        obs2 = torch.flatten(obs[1:, ...], 0, 1)
        actions = torch.flatten(actions[:-1, ...], 0, 1)

        loss = self.episodic_novelty.update(obs1, obs2, actions)
        return loss

    def train_lifelong_step(self, obs):
        obs = torch.flatten(obs, 0, 1)

        loss = self.lifelong_novelty.update(obs)
        return loss

    @staticmethod
    def soft_update(target, source, tau):
        """
        Soft weight updates: target slowly track the weights of source with constant tau
        See DDPG paper page 4: https://arxiv.org/pdf/1509.02971.pdf

        Parameters:
        target (nn.Module): target model
        source (nn.Module): source model
        tau (float): soft update constant
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )

    @staticmethod
    def hard_update(target, source):
        """
        Copy weights from source to target

        Parameters:
        target (nn.Module): target model
        source (nn.Module): source model
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

