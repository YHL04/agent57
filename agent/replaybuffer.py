

import torch
import threading
import time
import random
import numpy as np
from dataclasses import dataclass
from typing import List

from .logger import Logger


@dataclass
class Episode:
    """
    Episode dataclass used to store completed episodes from actor
    """
    obs: np.array
    actions: np.array
    rewards: np.array
    states: np.array
    length: int
    total_reward: float
    total_time: float


@dataclass
class Block:
    """
    Block dataclass used to store preprocessed batches for training
    """
    obs: torch.tensor
    actions: torch.tensor
    rewards: torch.tensor
    states: torch.tensor
    dones: torch.tensor
    idxs: List[List[int]]


class ReplayBuffer:
    """
    Replay Buffer will be used inside Learner where start_threads is called
    before the main training the loop. The Learner will asynchronously queue
    Episodes into the buffer, log the data, and prepare Block for training.

    Parameters:
    buffer_size (int): Size of self.buffer
    batch_size (int): Training batch size
    block (int): Time step length of blocks
    d_model (int): Dimension of model
    n_step (int): N step returns
    gamma (float): gamma constant for next q in q learning
    sample_queue (mp.Queue): FIFO queue to store Episode into ReplayBuffer
    batch_queue (mp.Queue): FIFO queue to sample batches for training from ReplayBuffer
    priority_queue (mp.Queue): FIFO queue to update new recurrent states from training to ReplayBuffer

    """

    def __init__(self,
                 buffer_size,
                 batch_size,
                 block,
                 n_step,
                 gamma,
                 sample_queue,
                 batch_queue,
                 priority_queue
                 ):

        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.block = block
        self.n_step = n_step

        self.gamma = np.full(n_step, gamma)**(np.arange(n_step))

        self.lock = threading.Lock()
        self.sample_queue = sample_queue
        self.batch_queue = batch_queue
        self.priority_queue = priority_queue

        self.buffer = np.empty(shape=(buffer_size,), dtype=object)
        self.logger = Logger()

        self.size = 0
        self.ptr = 0

    def __len__(self):
        return self.size

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

            if not self.batch_queue.full() and self.size != 0:
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
            self.buffer[self.ptr] = episode

            # increment size
            self.size += 1
            self.size = min(self.size, self.buffer_size)

            # increment pointer
            self.ptr += 1
            self.ptr = self.ptr % self.buffer_size

            # log
            self.logger.total_frames += episode.length
            self.logger.reward = episode.total_reward

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
            rewards = []
            states = []
            dones = []
            idxs = []

            for _ in range(self.batch_size):
                buffer_idx = random.randrange(0, self.size)
                time_idx = random.randrange(0, self.buffer[buffer_idx].length-self.n_step-self.block+1)
                idxs.append([buffer_idx, time_idx])

                rewards.append([
                    self.buffer[buffer_idx].rewards[time_idx+t:time_idx+t+self.n_step]
                    for t in range(self.block)
                ])
                obs.append(self.buffer[buffer_idx].obs[time_idx:time_idx+self.block+self.n_step])
                actions.append(self.buffer[buffer_idx].actions[time_idx:time_idx+self.block+self.n_step])
                states.append(torch.tensor(self.buffer[buffer_idx].states[time_idx]))
                dones.append(True if time_idx==self.buffer[buffer_idx].length-self.n_step-self.block
                             else False)

            obs = torch.tensor(np.stack(obs), dtype=torch.float32) / 255.
            actions = torch.tensor(np.stack(actions), dtype=torch.int32)
            rewards = torch.tensor(np.sum(np.array(rewards) * self.gamma, axis=2), dtype=torch.float32)
            states = None
            dones = torch.tensor(dones, dtype=torch.bool)

            obs = obs.transpose(0, 1)
            actions = actions.transpose(0, 1).unsqueeze(-1)
            rewards = rewards.transpose(0, 1).unsqueeze(-1)
            # states = states.to(torch.float32)
            dones = dones.unsqueeze(-1)

            assert obs.shape == (self.block+self.n_step, self.batch_size, 4, 105, 80)
            assert actions.shape == (self.block+self.n_step, self.batch_size, 1)
            assert rewards.shape == (self.block, self.batch_size, 1)
            # assert states[0].size == (self.batch_size, 512) and states[1].size == (self.batch_size == 512)
            assert dones.shape == (self.batch_size, 1)

            block = Block(obs=obs,
                          actions=actions,
                          rewards=rewards,
                          states=states,
                          dones=dones,
                          idxs=idxs
                          )

        return block

    def update_priorities(self, idxs, states, loss, epsilon):
        """
        Update recurrent states from new recurrent states obtained during training
        with most up-to-date model weights

        Parameters:
        idxs (List[List[buffer_idx, time_idx]]): indices of states
        states (Array[batch_size, block+n_step, state_len, d_model]): new recurrent states
        loss (float): critic loss
        bert_loss (float): bert loss
        epsilon (float): epsilon of Learner for logging purposes

        """
        # assert states.shape[0] == self.batch_size
        # assert states.shape[1] == self.block+self.n_step

        with self.lock:

            # update new state for each sample in batch
            # for idx, state in zip(idxs, states):
            #     buffer_idx, time_idx = idx

            #     self.buffer[buffer_idx].states[time_idx:time_idx+self.block+self.n_step] = state

            # log
            self.logger.total_updates += 1
            self.logger.loss = loss
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
        self.reward_buffer = []
        self.state_buffer = []

    def add(self, obs, action, reward, state):
        """
        This function is called after every time step to store data into list

        Parameters:
        alloc (float): allocation value
        timestep (datetime.datetime): timestamp of current time step
        action (float): recorded action
        reward (float): recorded reward
        state (Array[1, state_len, d_model]): recurrent state before model newly generated recurrent state
        """
        self.obs_buffer.append(obs)
        self.action_buffer.append(action)
        self.reward_buffer.append(reward)
        self.state_buffer.append(state)

    def finish(self, total_reward, total_time):
        """
        This function is called after episode ends. lists are
        converted into numpy arrays and lists are cleared for
        next episode

        Parameters:
        tickers (List[2]): List of tickers e.g. ["AAPL", "GOOGL"]
        total_reward (float): normalized total reward for benchmarking
        total_time (float): total time for actor to complete episode in seconds

        """
        obs = np.stack(self.obs_buffer).astype(np.uint8)
        actions = np.stack(self.action_buffer).astype(np.int32)
        rewards = np.stack(self.reward_buffer).astype(np.float32)
        states = np.stack(self.state_buffer)

        length = len(obs)

        self.obs_buffer.clear()
        self.action_buffer.clear()
        self.reward_buffer.clear()
        self.state_buffer.clear()

        return Episode(obs=obs,
                       actions=actions,
                       rewards=rewards,
                       states=states,
                       length=length,
                       total_reward=total_reward,
                       total_time=total_time
                       )
