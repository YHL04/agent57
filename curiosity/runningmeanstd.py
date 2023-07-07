

import numpy as np


class RunningMeanStd:
    """
    Modified from
    https://github.com/michaelnny/deep_rl_zoo/blob/main/deep_rl_zoo/normalizer.py
    for episodic novelty.
    """

    def __init__(self, shape=()):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = 0

        self.deltas = []
        self.min_size = 10

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)

        # update count and moments
        n = x.shape[0]
        self.count += n
        delta = batch_mean - self.mean
        self.mean += delta * n / self.count
        m_a = self.var * (self.count - n)
        m_b = batch_var * n
        M2 = m_a + m_b + np.square(delta) * self.count * n / self.count
        self.var = M2 / self.count

    def update_single(self, x):
        self.deltas.append(x)

        if len(self.deltas) >= self.min_size:
            batched_x = np.stack(self.deltas, axis=0)
            self.update(batched_x)

            del self.deltas[:]

    def normalize(self, x):
        return (x - self.mean) / np.sqrt(self.var + 1e-8)

