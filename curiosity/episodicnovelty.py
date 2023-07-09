

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
                 action_size,
                 N=10,
                 lr=5e-4,
                 kernel_epsilon=0.0001,
                 cluster_distance=0.008,
                 max_similarity=8.0,
                 c_constant=0.001,
                 device="cuda"
                 ):

        # dimension is always 512
        model = EmbeddingNet(action_size=action_size).to(device)

        self.model = deepcopy(model)
        self.eval_model = deepcopy(model)

        self.opt = optim.Adam(self.model.parameters(), lr=lr)

        self.index = faiss.IndexFlatL2(512)
        self.normalizer = RunningMeanStd()

        self.N = N
        self.kernel_epsilon = kernel_epsilon
        self.cluster_distance = cluster_distance
        self.max_similarity = max_similarity
        self.c_constant = c_constant

        self.count = 0

    def reset(self):
        self.index.reset()
        self.count = 0

    def add(self, emb):
        self.index.add(emb)
        self.count += 1

    def knn_query(self, emb):
        distance, _ = self.index.search(emb, self.N)
        return distance

    def get_reward(self, obs):
        emb = self.eval_model(obs).cpu().numpy()

        if self.count < self.N:
            self.add(emb)
            return 0.

        dist = self.knn_query(emb).squeeze()
        self.normalizer.update(dist)
        self.add(emb)

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

        if np.isnan(similarity) or similarity > self.max_similarity:
            return 0.

        return (1 / similarity).item()

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

