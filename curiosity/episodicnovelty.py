

import torch
import torch.optim as optim
import torch.nn.functional as F
import faiss

from copy import deepcopy

from models import EmbeddingNet
from .runningmeanstd import RunningMeanStd


class EpisodicNovelty:

    def __init__(self,
                 N=5,
                 lr=5e-4,
                 kernel_epsilon=0.0001,
                 cluster_distance=0.008,
                 max_similarity=8.0,
                 c_constant=0.001
                 ):

        # dimension is always 512
        model = EmbeddingNet()

        self.model = deepcopy(model)
        self.eval_model = deepcopy(model)

        self.opt = optim.Adam(self.model.parameters(), lr=lr)

        self.index = faiss.IndexFlatL2(512, N)
        self.normalizer = RunningMeanStd()

        self.N = N
        self.kernel_epsilon = kernel_epsilon
        self.cluster_distance = cluster_distance
        self.max_similarity = max_similarity
        self.c_constant = c_constant

    def reset(self):
        self.index.reset()

    def add(self, emb):
        self.index.add(emb)

    def knn_query(self, emb):
        distance, _ = self.index.search(emb, self.N)
        return distance

    def get_reward(self, obs):
        emb = self.eval_model(obs).squeeze(0)

        dist = self.knn_query(emb)
        self.add(emb)

        # update running mean
        self.normalizer.update_single(dist)

        # normalize distances with running mean
        distance_rate = dist / (self.normalizer.mean + 1e-8)

        # the distance becomes 0 if already small
        distance_rate = torch.min((distance_rate - self.cluster_distance), torch.tensor(0.))

        # compute the kernel value
        kernel_output = self.kernel_epsilon / (distance_rate + self.kernel_epsilon)

        # similarity
        similarity = torch.sqrt(torch.sum(kernel_output)) + self.c_constant

        if torch.isnan(similarity):
            return 0.

        if similarity > self.max_similarity:
            return 0.

        return (1 / similarity).cpu().item()

    def update(self, obs1, obs2, actions):
        emb1 = self.model.forward(obs1)
        emb2 = self.model.forward(obs2)
        emb = torch.concat([emb1, emb2], dim=-1)
        logits = self.model.inverse(emb)

        self.opt.zero_grad()
        loss = F.cross_entropy(logits, actions)
        loss = loss.mean()
        self.opt.step()

        return loss.item()

    def update_eval(self):
        self.eval_model.load_state_dict(self.model.state_dict())

