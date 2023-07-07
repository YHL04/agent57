

import torch
import faiss


class EpisodicNovelty:

    def __init__(self,
                 model,
                 capacity,
                 num_neighbors,
                 kernel_epsilon=0.0001,
                 cluster_distance=0.008,
                 max_similarity=8.0,
                 c_constant=0.001
                 ):

        self.model = model
        self.buffer = None

        self.capacity = capacity

        self.num_neighbors = num_neighbors
        self.kernel_epsilon = kernel_epsilon
        self.cluster_distance = cluster_distance
        self.max_similarity = max_similarity
        self.c_constant = c_constant

    def reset(self):
        self.memory.clear()

    def add(self, emb):
        self.memory.add(emb)

    def knn_query(self, emb):
        vectors = None
        dist = None

        return vectors, dist

    def compute_bonus(self, obs):
        emb = self.model(obs).squeeze(0)
        self.add(emb)

        vectors, dist = self.knn_query(emb, self.num_neighbors)

        # update running mean
        self.normalizer.update(dist)

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

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

