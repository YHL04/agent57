

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch.multiprocessing as mp
import torch.distributed.rpc as rpc
from torch.distributed.rpc import RRef
from torch.distributed.rpc.functions import async_execution
from torch.futures import Future

import numpy as np
import threading
import time
import random
from copy import deepcopy

from .actor import Actor
from .replay_buffer import ReplayBuffer

from model import Model


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
    burnin_len (int): Length of burnin, concept from R2D2 paper
    rollout_len (int): Length of rollout, concept from R2D2 paper

    """
    epsilon = 1
    epsilon_min = 0.2
    epsilon_decay = 0.0001

    lr = 1e-4
    gamma = 0.99

    tau = 0.01
    save_every = 100

    def __init__(self,
                 model,
                 buffer_size,
                 batch_size,
                 vocab_size,
                 max_len,
                 d_model,
                 n_tau,
                 n_p,
                 state_len,
                 n_step,
                 burnin_len,
                 rollout_len
                 ):
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.n_tau = n_tau

        # models
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
        self.burnin_len = burnin_len
        self.rollout_len = rollout_len
        self.block_len = burnin_len + rollout_len
        self.n_step = n_step
        self.gamma = self.gamma ** n_step

        # optimizer and loss functions
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

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
                                          block_len=burnin_len+rollout_len,
                                          n_step=n_step,
                                          gamma=self.gamma,
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
                                           tickers=self.tickers,
                                           d_model=self.d_model,
                                           state_len=self.state_len
                                           )

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

        return future

    def prepare_data(self):
        """
        Thread to prepare batch for update, batch_queue is filled by ReplayBuffer
        Loops through with a time gap of 0.1 sec
        """

        while True:
            time.sleep(0.1)

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

        time.sleep(2)
        while True:
            time.sleep(1)

            while not self.batch_data:
                time.sleep(0.1)
            block = self.batch_data.pop(0)

            self.update(obs=block.obs,
                        actions=block.actions,
                        rewards=block.rewards,
                        states=block.states,
                        idxs=block.idxs
                        )

    def update(self, obs, actions, rewards, bert_targets, states, idxs):
        """
        An update step. Performs a training step, update new recurrent states,
        soft update target model and transfer weights to eval model
        """
        loss, new_states = self.train_step(obs=obs.cuda(),
                                           actions=actions.cuda(),
                                           rewards=rewards.cuda(),
                                           bert_targets=bert_targets.cuda(),
                                           states=states.cuda()
                                           )

        # update new states to buffer
        self.priority_queue.put((idxs, new_states, loss, self.epsilon))

        # soft update target model
        self.soft_update(self.target_model, self.model, self.tau)

        # transfer weights to eval model
        with self.lock_model:
            self.hard_update(self.eval_model, self.model)

        return loss

    def train_step(self, obs, actions, rewards, states):
        """
        Accumulate gradients to increase batch size
        Gradients are cached for n_accumulate steps before optimizer.step()

        Parameters:
        allocs (Tensor[block_len+n_step, batch_size*n_accumulate, 1]): allocation values
        ids (Tensor[block_len+n_step, batch_size*n_accumulate, max_len]): tokens
        actions (Tensor[block_len+n_step, batch_size*n_accumulate, 1, 1]): recorded actions
        rewards (Tensor[block_len, batch_size*n_accumulate, 1]): recorded rewards
        bert_targets (Tensor[block_len+n_step, batch_size*n_accumulate, 1]): bert targets
        states (Tensor[batch_size*n_accumulate, state_len, d_model]): recorded recurrent states

        Returns:
        loss (float): Loss of critic model
        bert_loss (float): Loss of bert masked language modeling
        new_states (float): Generated new states with new weights during training

        """

        with torch.no_grad():
            state = states.detach().clone()

            new_states = []
            for t in range(self.burnin+self.n_step):
                new_states.append((state[0].detach(), state[1].detach()))
                _, state = self.target_model.forward(obs[t], state)

            next_q = []
            for t in range(self.burnin+self.n_step, self.block+self.n_step):
                new_states.append((state[0].detach(), state[1].detach()))
                next_q_, state = self.target_model.forward(obs[t], state)
                next_q.append(next_q_)

            next_q = torch.stack(next_q)
            next_q = torch.max(next_q, axis=-1, keepdim=True)[0].to(torch.float32)

            next_q = rewards[self.burnin:] + self.gamma * next_q
            assert next_q.shape == (self.rollout, self.bsz, 1)

        self.model.zero_grad()

        state = new_states[self.burnin].detach()
        expected = []
        target = []
        for t in range(self.burnin, self.block):
            expected_, state = self.model(obs[t], state)
            expected.append(expected_)

            target_ = expected_.detach()
            target_[torch.arange(self.bsz), actions[t]] = next_q[t-self.burnin]
            target.append(target_.detach())

        expected = torch.stack(expected)
        target = torch.stack(target)
        loss = F.huber_loss(expected, target)
        loss.backward()
        self.opt.step()

        loss = loss.item()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
        self.optimizer.step()

        return loss, new_states

