

import torch
import threading
import time
import random
import numpy as np
from dataclasses import dataclass
from typing import List
from collections import deque

from .logger import Logger
from .per import SumTree


@dataclass
class Episode:
    """
    Episode dataclass used to store completed episodes from actor
    """
    obs: np.array
    actions: np.array
    probs: np.array
    extr: np.array
    intr: np.array
    states1: np.array
    states2: np.array
    dones: np.array
    sumtree: SumTree
    length: int
    beta: float
    discount: float
    total_extr: float
    total_intr: float
    total_time: float
    signature: str


@dataclass
class Block:
    """
    Block dataclass used to store preprocessed batches for training
    """
    obs: torch.tensor
    actions: torch.tensor
    probs: torch.tensor
    extr: torch.tensor
    intr: torch.tensor
    states1: torch.tensor
    states2: torch.tensor
    dones: torch.tensor
    discounts: torch.tensor
    is_weights: torch.tensor
    idxs: List[List[int]]


class ReplayBuffer:
    """
    Replay Buffer will be used inside Learner where start_threads is called
    before the main training the loop. The Learner will asynchronously queue
    Episodes into the buffer, log the data, and prepare Block for training.

    Args:
        size (int): Size of self.buffer
        B (int): Training batch size
        T (int): Time step length of blocks
        discount (float): Gamma constant for next q in q learning
        beta (float): Maximum beta value in Never Give Up agent
        sample_queue (mp.Queue): FIFO queue to store Episode into ReplayBuffer
        batch_queue (mp.Queue): FIFO queue to sample batches for training from ReplayBuffer
        priority_queue (mp.Queue): FIFO queue to update new recurrent states from training to ReplayBuffer

    """
    p_a = 0.6
    p_beta = 0.4
    p_beta_increment_per_sampling = 0.001

    def __init__(self, size, B, burnin, rollout, T, beta,
                 sample_queue, batch_queue, priority_queue
                 ):

        self.size = size
        self.B = B
        self.burnin = burnin
        self.rollout = rollout
        self.T = T
        self.beta = beta

        self.lock = threading.Lock()
        self.sample_queue = sample_queue
        self.batch_queue = batch_queue
        self.priority_queue = priority_queue

        # Buffer
        self.buffer = np.empty((size,), dtype=object)
        self.ptr, self.count, self.n_entries = 0, 0, 0

        # Global Sum Tree
        self.sumtree = SumTree(size, max_error=0, fill_zero=True)

        # Logger
        self.logger = Logger()

        self.frames = 0
        self.max_error = 1
        self.total_p = 0

    def __len__(self):
        return self.count

    def start_threads(self):
        """Wrapper function to start all the threads in ReplayBuffer"""
        thread = threading.Thread(target=self.add_data, daemon=True)
        thread.start()

        thread = threading.Thread(target=self.prepare_data, daemon=True)
        thread.start()

        thread = threading.Thread(target=self.update_data, daemon=True)
        thread.start()

        thread = threading.Thread(target=self.log_data, daemon=True)
        thread.start()

    def add_data(self):
        """asynchronously add episodes to buffer by calling add()"""
        while True:
            time.sleep(0.001)

            if not self.sample_queue.empty():
                data = self.sample_queue.get_nowait()
                self.add(data)

    def prepare_data(self):
        """asynchronously add batches to batch_queue by calling sample_batch()"""
        while True:
            time.sleep(0.001)

            if not self.batch_queue.full() and self.count != 0:
                data = self.sample_batch()
                self.batch_queue.put(data)

    def update_data(self):
        """asynchronously update states inside buffer by calling update_priorities()"""
        while True:
            time.sleep(0.001)

            if not self.priority_queue.empty():
                data = self.priority_queue.get_nowait()
                self.update_priorities(*data)

    def log_data(self):
        """asynchronously prints out logs and write into file by calling log()"""
        while True:
            time.sleep(10)

            self.log()

    def add(self, episode):
        """Add Episode to self.buffer and update size, ptr, and log"""

        with self.lock:

            # add to buffer
            self.frames += episode.length
            # create sum tree for priority queue
            episode.sumtree = SumTree(episode.length-self.T, self.max_error, fill_zero=False)
            # subtract entries that will be removed along with episode
            if self.buffer[self.ptr] is not None:
                self.n_entries -= episode.length-self.T
            # append episode to buffer
            self.buffer[self.ptr] = episode
            # update global sumtree
            assert episode.sumtree.total() != 0
            self.sumtree.add(self.ptr, episode.sumtree.total(), got_p=True)

            # increment pointers and counts
            self.n_entries += episode.length-self.T

            self.count = min(self.count + 1, self.size)
            self.ptr += 1
            if self.ptr >= self.size:
                self.ptr = 0

            # log
            self.logger.total_frames += episode.length
            self.logger.beta = episode.beta

            # obtain extrinsic reward from purely exploitative policy
            if episode.beta == 0:
                self.logger.reward = episode.total_extr

            # obtain intrinsic reward from purely exploration policy
            if episode.beta == self.beta:
                self.logger.intrinsic = episode.total_intr

    def sample_batch(self):
        """
        Sample batch from buffer by sampling allocs, ids, actions, rewards, states, idxs.
        Then create bert targets from ids and precompute rewards with n step and gamma.
        Finally return finished Block for training.

        Returns:
            block (Block): completed block

        """

        with self.lock:

            obs = []
            actions = []
            probs = []
            extr = []
            intr = []
            states1 = []
            states2 = []
            dones = []
            discounts = []
            priorities = []
            idxs = []

            # increment per constants
            self.p_beta = min(1., self.p_beta + self.p_beta_increment_per_sampling)

            total = self.sumtree.total()
            segment = total / self.B

            for i in range(self.B):
                # normal experience replay
                # b_idx = random.randrange(0, self.count)
                # t_idx = random.randrange(0, self.buffer[b_idx].length-self.T)

                # sample from prioritized experience replay
                a = segment * i
                b = segment * (i + 1)
                s = random.uniform(a, b)

                b_idx, p, s = self.sumtree.get(s)
                t_idx, p, _ = self.buffer[b_idx].sumtree.get(s)

                priorities.append(p)
                idxs.append([b_idx, t_idx, self.buffer[b_idx].signature])

                extr.append(self.buffer[b_idx].extr[t_idx:t_idx+self.T])
                intr.append(self.buffer[b_idx].intr[t_idx:t_idx+self.T])
                obs.append(self.buffer[b_idx].obs[t_idx:t_idx+self.T+1])
                actions.append(self.buffer[b_idx].actions[t_idx:t_idx+self.T+1])
                probs.append(self.buffer[b_idx].probs[t_idx:t_idx+self.T+1])
                states1.append(self.buffer[b_idx].states1[t_idx])
                states2.append(self.buffer[b_idx].states2[t_idx])
                dones.append(self.buffer[b_idx].dones[t_idx:t_idx+self.T])
                discounts.append(self.buffer[b_idx].discount)

            obs = torch.tensor(np.stack(obs), dtype=torch.float32) / 255.
            actions = torch.tensor(np.stack(actions), dtype=torch.int32)
            probs = torch.tensor(np.stack(probs), dtype=torch.float32)

            extr = torch.tensor(np.stack(extr), dtype=torch.float32)
            intr = torch.tensor(np.stack(intr), dtype=torch.float32)

            states1 = torch.tensor(np.stack(states1), dtype=torch.float32)
            states2 = torch.tensor(np.stack(states2), dtype=torch.float32)
            states1 = (states1[:, 0, :], states1[:, 1, :])
            states2 = (states2[:, 0, :], states2[:, 1, :])

            dones = torch.tensor(np.stack(dones), dtype=torch.bool)
            discounts = torch.tensor(np.stack(discounts), dtype=torch.float32)

            obs = obs.transpose(0, 1)
            actions = actions.transpose(0, 1)
            probs = probs.transpose(0, 1)
            extr = extr.transpose(0, 1)
            intr = intr.transpose(0, 1)
            dones = dones.transpose(0, 1)

            # prioritized experience replay
            priorities = torch.tensor(priorities) ** self.p_a
            sampling_probabilities = priorities / priorities.sum()
            is_weights = torch.pow(self.n_entries * sampling_probabilities, -self.p_beta)
            is_weights /= is_weights.max()
            assert torch.isnan(is_weights).any() == False

            assert obs.shape == (self.T+1, self.B, 4, 105, 80)
            assert actions.shape == (self.T+1, self.B)
            assert probs.shape == (self.T+1, self.B)
            assert extr.shape == (self.T, self.B)
            assert intr.shape == (self.T, self.B)
            assert states1[0].shape == (self.B, 512) and states1[1].shape == (self.B, 512)
            assert states2[0].shape == (self.B, 512) and states2[1].shape == (self.B, 512)
            assert dones.shape == (self.T, self.B)
            assert discounts.shape == (self.B,)
            assert is_weights.shape == (self.B,)

            block = Block(obs=obs,
                          actions=actions,
                          probs=probs,
                          extr=extr,
                          intr=intr,
                          states1=states1,
                          states2=states2,
                          dones=dones,
                          discounts=discounts,
                          is_weights=is_weights,
                          idxs=idxs
                          )

        return block

    def update_priorities(self, idxs, states1, states2, errors, loss, intr_loss, epsilon):
        """
        Update recurrent states from new recurrent states obtained during training
        with most up-to-date model weights. Data are the training batch

        Args:
            idxs (List[List[b_idx, t_idx]]): indices of states
        """
        assert states1.shape == (self.B, self.T+1, 2, 512)
        assert states2.shape == (self.B, self.T+1, 2, 512)
        assert errors.shape == (self.B,)

        with self.lock:

            for (idx, state1, state2, error) in zip(idxs, states1, states2, errors):
                # b_idx is the index of episode and t_idx is the starting index of burnin
                b_idx, t_idx, signature = idx

                if signature != self.buffer[b_idx].signature:
                    continue

                self.buffer[b_idx].sumtree.update(t_idx, error, got_p=False)
                self.sumtree.update(b_idx, self.buffer[b_idx].sumtree.total(), got_p=True)

                if error > self.max_error:
                    self.max_error = error

            # update new state for each sample in batch
            # for idx, state1, state2 in zip(idxs, states1, states2):
            #     b_idx, t_idx = idx
            #
            #     try:
            #         self.buffer[b_idx].states1[t_idx:t_idx+self.T+1] = state1
            #         self.buffer[b_idx].states2[t_idx:t_idx+self.T+1] = state2
            #
            #     except IndexError:
            #         pass
            #
            #     except ValueError:
            #         pass

            # log
            self.logger.total_updates += 1
            self.logger.loss = loss
            self.logger.intr_loss = intr_loss
            self.logger.epsilon = epsilon

    def log(self):
        """
        Calls logger.print() to print out all the tracked values during training,
        lock to make sure its thread safe
        """

        with self.lock:
            self.logger.print()


class LocalBuffer:
    """
    Used by Actor to store data. Once the episode is finished
    finish() is called to return Episode to Learner to store in ReplayBuffer
    """

    def __init__(self):
        self.obs_buffer = []
        self.action_buffer = []
        self.prob_buffer = []
        self.extr_buffer = []
        self.intr_buffer = []
        self.state1_buffer = []
        self.state2_buffer = []

    def add(self, obs, action, prob, extr, intr, state1, state2):
        """
        This function is called after every time step to store data into list

        Args:
            obs (Array): observed frame
            action (float): recorded action
            reward (float): recorded reward
            state (Array): recurrent state before model newly generated recurrent state
        """
        self.obs_buffer.append(obs)
        self.action_buffer.append(action)
        self.prob_buffer.append(prob)
        self.extr_buffer.append(extr)
        self.intr_buffer.append(intr)
        self.state1_buffer.append(state1)
        self.state2_buffer.append(state2)

    def finish(self, total_time, beta, discount, signature):
        """
        This function is called after episode ends. lists are
        converted into numpy arrays and lists are cleared for
        next episode

        Args:
            total_time (float): total time for actor to complete episode in seconds
            beta (float): beta associated with that episode
            discount (float): discount associated with that episode
            signature (string): a unique signature associated to each episode

        """

        # pad obs, action, prob for retrace so that done can be included
        # since last element is not needed for retrace loss
        self.obs_buffer.append(np.zeros_like(self.obs_buffer[-1]))
        self.action_buffer.append(np.zeros_like(self.action_buffer[-1]))
        self.prob_buffer.append(np.zeros_like(self.prob_buffer[-1]))

        obs = np.stack(self.obs_buffer).astype(np.uint8)
        actions = np.stack(self.action_buffer).astype(np.int32)
        probs = np.stack(self.prob_buffer).astype(np.float32)
        extr = np.stack(self.extr_buffer).astype(np.float32)
        intr = np.stack(self.intr_buffer).astype(np.float32)
        states1 = np.stack(self.state1_buffer).astype(np.float32)
        states2 = np.stack(self.state2_buffer).astype(np.float32)

        dones = np.zeros_like(extr)
        dones[-1] = 1
        dones = dones.astype(np.bool)

        length = len(extr)

        total_extr = np.sum(extr).item()
        total_intr = np.sum(intr).item()

        self.obs_buffer.clear()
        self.action_buffer.clear()
        self.prob_buffer.clear()
        self.extr_buffer.clear()
        self.intr_buffer.clear()
        self.state1_buffer.clear()
        self.state2_buffer.clear()

        return Episode(obs=obs,
                       actions=actions,
                       probs=probs,
                       extr=extr,
                       intr=intr,
                       states1=states1,
                       states2=states2,
                       dones=dones,
                       sumtree=None,
                       length=length,
                       beta=beta,
                       discount=discount,
                       total_extr=total_extr,
                       total_intr=total_intr,
                       total_time=total_time,
                       signature=signature
                       )
