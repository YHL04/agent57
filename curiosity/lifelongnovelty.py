

import torch
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

from models import ConvNet
from .runningmeanstd import RunningMeanStd


class LifelongNovelty:

    def __init__(self,
                 lr=5e-4,
                 L=5,
                 device="cuda"):

        self.predictor = ConvNet().to(device)
        self.target = ConvNet().to(device)

        self.eval_predictor = ConvNet().to(device)
        self.eval_target = ConvNet().to(device)

        self.opt = optim.Adam(self.predictor.parameters(), lr=lr)

        self.normalizer = RunningMeanStd()

        self.L = L
        self.device = device

    def normalize_reward(self, reward):
        """Compute returns then normalize the intrinsic reward based on these returns"""

        self.normalizer.update_single(reward)

        norm_reward = reward / np.sqrt(self.normalizer.var + 1e-8)
        return norm_reward.item()

    def get_reward(self, obs):

        # ngu paper normalizes obs but we dont since its already 0-1
        pred = self.predictor(obs)
        target = self.target(obs)

        reward = torch.square(pred - target).mean().detach().cpu().item()
        reward = self.normalize_reward(reward)
        reward = min(max(reward, 1), self.L)

        return reward

    def update(self, obs):
        target = self.predictor(obs)
        expected = self.target(obs)

        self.opt.zero_grad()
        loss = F.mse_loss(expected, target)
        loss = loss.mean()
        self.opt.step()

        return loss.item()

    def update_eval(self):
        self.eval_predictor.load_state_dict(self.predictor.state_dict())
        self.eval_target.load_state_dict(self.target.state_dict())

