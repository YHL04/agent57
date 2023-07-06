

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLSTM(nn.Module):

    def __init__(self, action_size):
        super(ConvLSTM, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(64, 1024, kernel_size=7, stride=1)
        self.fc = nn.Linear(7 * 7 * 64, 512)

        self.lstm = nn.LSTMCell(512, 512)
        self.out = nn.Linear(512, action_size)

    def forward(self, x, state):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)

        x = F.relu(self.fc(x))
        x, state = self.lstm(x, state)
        state = (x, state)
        x = self.out(x)

        return x, state

