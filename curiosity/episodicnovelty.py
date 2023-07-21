

import torch
import torch.optim as optim
import torch.nn.functional as F

import faiss
import numpy as np
from copy import deepcopy

from models import EmbeddingNet
from .runningmeanstd import RunningMeanStd


class EpisodicNovelty:
    """
    Different from original implementation, index resets after n timesteps
    """

    def __init__(self,
                 num_envs,
                 action_size,
                 N=10,
                 lr=5e-4,
                 kernel_epsilon=0.0001,
                 cluster_distance=0.008,
                 max_similarity=8.0,
                 c_constant=0.001,
                 device="cuda"
                 ):

        self.num_envs = num_envs

        # dimension is always 512
        model = EmbeddingNet(action_size=action_size).to(device)

        self.model = deepcopy(model)
        self.eval_model = deepcopy(model)

        self.opt = optim.Adam(self.model.parameters(), lr=lr)

        self.index = [faiss.IndexFlatL2(512) for _ in range(num_envs)]
        self.normalizer = RunningMeanStd()

        self.N = N
        self.kernel_epsilon = kernel_epsilon
        self.cluster_distance = cluster_distance
        self.max_similarity = max_similarity
        self.c_constant = c_constant

        self.counts = torch.zeros((num_envs,))

    def reset(self, ids):
        for id in ids:
            self.index[id].reset()
            self.counts[id] = 0

    def add(self, ids, emb):
        for i, id in enumerate(ids):
            self.index[id].add(emb[i].numpy())
            self.counts[id] += 1

    def knn_query(self, ids, emb):
        distances = []
        for id in ids:
            distance, _ = self.index[id].search(emb.numpy(), self.N)
            distances.append(distance)

        return torch.tensor(np.stack(distances))

    def get_reward(self, ids, obs):
        emb = self.eval_model(obs)

        # if self.counts[id] < self.N:
        #     self.add(id, emb)
        #     return 0.

        dist = self.knn_query(ids, emb)
        self.normalizer.update(dist)
        self.add(ids, emb)

        # Calculate kernel output
        # print('dist ', dist.mean())
        # print('mean ', self.normalizer.mean)
        distance_rate = dist / (self.normalizer.mean + 1e-8)
        # print('should average 1 ', distance_rate)

        distance_rate = np.maximum((distance_rate - self.cluster_distance), np.array(0.))
        kernel_output = self.kernel_epsilon / (distance_rate + self.kernel_epsilon)

        # Calculate denominator
        # different from original paper: mean instead of sum so scale is independent of self.N
        similarity = np.sqrt(np.sum(kernel_output)) + self.c_constant
        # print('sim ', similarity)
        # if similarity < 1:
        #     print('dst', distance_rate)

        mask = (self.counts < self.N) | torch.isnan(similarity) | (similarity > self.max_similarity)
        intr = torch.where(mask, 0., 1 / similarity)

        return intr

    def update(self, obs1, obs2, actions):
        emb1 = self.model.forward(obs1)
        emb2 = self.model.forward(obs2)
        emb = torch.concat([emb1, emb2], dim=-1)
        logits = self.model.inverse(emb)

        self.opt.zero_grad()
        loss = F.cross_entropy(logits, actions.to(torch.int64).squeeze())
        loss = loss.mean()
        self.opt.step()

        return loss.item()

    def update_eval(self):
        self.eval_model.load_state_dict(self.model.state_dict())

