

import torch
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

from models import ConvNet
from runningmeanstd import RunningMeanStd


class LifelongNovelty:

    def __init__(self,
                 lr=5e-4):

        self.predictor = ConvNet()
        self.target = ConvNet()

        self.opt = optim.Adam(self.predictor.parameters(), lr=lr)

        self.normalizer = RunningMeanStd()

    def normalize_reward(self, reward):
        """Compute returns then normalize the intrinsic reward based on these returns"""

        self.normalizer.update_single(reward)

        norm_reward = reward / np.sqrt(self.normalizer.var + 1e-8)
        return norm_reward.item()

    def get_reward(self, obs):
        norm_obs = self.normalize_obs(obs)

        pred = self.predictor(norm_obs)
        target = self.target(norm_obs)

        reward = torch.square(pred - target).mean(dim=1).detach().cpu().numpy()
        norm_reward = self.normalize_reward(reward)
        return norm_reward

    def update(self, obs):
        target = self.predictor(obs)
        expected = self.target(obs)

        self.opt.zero_grad()
        loss = F.mse_loss(expected, target)
        loss = loss.mean()
        self.opt.step()

        return loss.item()

