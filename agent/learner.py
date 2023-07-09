

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

from models import SingleModel, EmbeddingNet
from curiosity import EpisodicNovelty, LifelongNovelty


class Learner:
    """
    Main class used to train the agent. Called by rpc remote.
    Call run() to start the main training loop.

    Parameters:
    buffer_size (int): The size of the buffer in ReplayBuffer
    batch_size (int): Batch size for training
    n_layers (int): Number of layers in transformer
    n_cos (int): Number of cosine samples for each tau in IQN
    n_tau (int): Number of tau samples for IQN each representing a value for a percentile
    n_step (int): N step returns see https://paperswithcode.com/method/n-step-returns
    burnin (int): Length of burnin, concept from R2D2 paper
    rollout (int): Length of rollout, concept from R2D2 paper

    """
    epsilon = 1
    epsilon_min = 0.1
    epsilon_decay = 0.0001

    lr = 1e-4
    gamma = 0.95
    beta = 0.05  # 0.2

    update_every = 400
    save_every = 100
    device = "cuda"

    def __init__(self,
                 env_name,
                 buffer_size,
                 batch_size,
                 n_cos,
                 n_tau,
                 n_step,
                 burnin,
                 rollout
                 ):
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.n_tau = n_tau

        # models
        self.action_size = gym.make(env_name).action_space.n
        model = SingleModel(action_size=self.action_size)

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
        self.block = burnin + rollout
        self.n_step = n_step
        self.gamma = self.gamma ** n_step

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
        self.replay_buffer = ReplayBuffer(buffer_size=buffer_size,
                                          batch_size=batch_size,
                                          block=burnin+rollout,
                                          n_step=n_step,
                                          gamma=self.gamma,
                                          beta=self.beta,
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
    def get_policy(self, obs, state):
        self.epsilon -= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)

        with self.lock_model:
            q_values, state = self.eval_model(obs, state)
            intr_e = self.episodic_novelty.get_reward(obs)
            intr_l = self.lifelong_novelty.get_reward(obs)
            intrinsic = intr_e * intr_l

            state = (state[0].detach().cpu().numpy(),
                     state[1].detach().cpu().numpy())

        if random.random() <= self.epsilon:
            return random.randrange(0, self.action_size), state, intrinsic

        action = torch.argmax(q_values.squeeze()).detach().cpu().squeeze().item()
        return action, state, intrinsic

    def get_action(self, obs, state):
        if state is None:
            state = (np.zeros((1, 512)), np.zeros((1, 512)))

        obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0) / 255.
        state = (torch.tensor(state[0], dtype=torch.float32, device=self.device),
                 torch.tensor(state[1], dtype=torch.float32, device=self.device))

        action, state, intrinsic = self.get_policy(obs, state)
        return action, state, intrinsic

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
                        rewards=block.rewards,
                        states=block.states,
                        dones=block.dones,
                        idxs=block.idxs
                        )

    def update(self, obs, actions, rewards, states, dones, idxs):
        """
        An update step. Performs a training step, update new recurrent states,
        soft update target model and transfer weights to eval model
        """
        loss, new_states = self.train_step(obs=obs.cuda(),
                                           actions=actions.cuda(),
                                           rewards=rewards.cuda(),
                                           states=(states[0].cuda(), states[1].cuda()),
                                           dones=dones.cuda()
                                           )
        intr_loss = self.train_novelty_step(obs=obs.cuda(),
                                            actions=actions.cuda()
                                            )

        # reformat List[Tuple(Tensor, Tensor)] to array of shape (bsz, block_len+n_step, 2, dim)
        states1, states2 = zip(*new_states)
        states1 = torch.stack(states1).transpose(0, 1).cpu().numpy()
        states2 = torch.stack(states2).transpose(0, 1).cpu().numpy()
        new_states = np.stack([states1, states2], 2)

        # update new states to buffer
        self.priority_queue.put((idxs, new_states, loss, intr_loss, self.epsilon))

        # soft update target model
        if self.updates % self.update_every == 0:
            self.hard_update(self.target_model, self.model)

        # transfer weights to eval model
        with self.lock_model:
            self.hard_update(self.eval_model, self.model)
            self.episodic_novelty.update_eval()
            self.lifelong_novelty.update_eval()

        return loss, intr_loss

    def train_step(self, obs, actions, rewards, states, dones):
        """
        Accumulate gradients to increase batch size
        Gradients are cached for n_accumulate steps before optimizer.step()

        Parameters:
        allocs (Tensor[block+n_step, batch_size*n_accumulate, 1]): allocation values
        ids (Tensor[block+n_step, batch_size*n_accumulate, max_len]): tokens
        actions (Tensor[block+n_step, batch_size*n_accumulate, 1, 1]): recorded actions
        rewards (Tensor[block, batch_size*n_accumulate, 1]): recorded rewards
        bert_targets (Tensor[block+n_step, batch_size*n_accumulate, 1]): bert targets
        states (Tensor[batch_size*n_accumulate, state_len, d_model]): recorded recurrent states

        Returns:
        loss (float): Loss of critic model
        bert_loss (float): Loss of bert masked language modeling
        new_states (float): Generated new states with new weights during training

        """

        with torch.no_grad():
            state = (states[0].detach().clone(), states[1].detach().clone())

            new_states = []
            for t in range(self.burnin+self.n_step):
                new_states.append((state[0].detach(), state[1].detach()))
                _, state = self.target_model(obs[t], state)

            next_q = []
            for t in range(self.burnin+self.n_step, self.block+self.n_step):
                new_states.append((state[0].detach(), state[1].detach()))
                next_q_, state = self.target_model(obs[t], state)
                next_q.append(next_q_)

            next_q = torch.stack(next_q)
            next_q = torch.max(next_q, axis=-1, keepdim=True).values.to(torch.float32)

            # calculate next q values
            next_q = rewards[self.burnin:] + self.gamma * next_q

            # if done replace next_q with only rewards
            next_q[-1] = torch.where(dones, rewards[-1], next_q[-1])

            assert next_q.shape == (self.rollout, self.batch_size, 1)

        self.model.zero_grad()

        state = new_states[self.burnin]
        expected = []
        target = []
        for t in range(self.burnin, self.block):
            expected_, state = self.model(obs[t], state)
            expected.append(expected_)

            target_ = expected_.detach().clone()
            target_[torch.arange(self.batch_size), actions[t].squeeze()] = next_q[t-self.burnin].squeeze()
            target.append(target_)

        expected = torch.stack(expected)
        target = torch.stack(target)
        assert not torch.equal(expected, target)
        loss = F.huber_loss(expected, target)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        loss = loss.item()
        return loss, new_states

    def train_novelty_step(self, obs, actions):
        emb_loss = self.train_emb_step(obs, actions)
        lifelong_loss = self.train_lifelong_step(obs)

        return emb_loss + lifelong_loss

    def train_emb_step(self, obs, actions):
        """
        Args:
            obs (torch.Tensor): shape (block+n_step, bsz, 4, 105, 80)
            actions (torch.Tensor): shape (block+n_step, bsz, 1)
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

