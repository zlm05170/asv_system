import os
import gym
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class PolicyNet(nn.Module):
    def __init__(self, scan_frames, device="cpu"):
        super(PolicyNet, self).__init__()
        self.device = device

        # L_in = 72, L_out = 34
        self.conv1 = nn.Conv1d(scan_frames, 16, 5, stride=2)
        # L_in = 34, L_out = 15
        self.conv2 = nn.Conv1d(16, 16, 5, stride=2)
        # L_in = 15, L_out = 6
        self.conv3 = nn.Conv1d(16, 16, 5, stride=2)
        self.fc = nn.Linear(6*16+2+1+2+1, 64)
        # self.act = nn.Linear(64, 2)
        self.act = nn.Linear(64, 1)

    def forward(self, obs, state=None, info={}):
        scan = th.as_tensor(obs.scan, device=self.device, dtype=th.float32)
        position = th.as_tensor(obs.position, device=self.device, dtype=th.float32)
        heading = th.as_tensor(obs.heading, device=self.device, dtype=th.float32)
        goal = th.as_tensor(obs.goal, device=self.device, dtype=th.float32)
        lact = th.as_tensor(obs.laction, device=self.device, dtype=th.float32)

        x = F.relu(self.conv1(scan))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.shape[0], -1)
        x = th.cat((x, position, heading, goal, lact), dim=-1)
        x = F.relu(self.fc(x))
        # mu1 = 10. * th.sigmoid(self.act1(x))
        mu = np.pi/2 * th.tanh(self.act(x))
        # mu = th.cat((mu1, mu2), dim=-1)

        return mu, state

class ValueNet(nn.Module):
    def __init__(self, scan_frames, device="cpu"):
        super(ValueNet, self).__init__()
        self.device = device

        # L_in = 72, L_out = 34
        self.conv1 = nn.Conv1d(scan_frames, 16, 5, stride=2)
        # L_in = 34, L_out = 15
        self.conv2 = nn.Conv1d(16, 16, 5, stride=2)
        # L_in = 15, L_out = 6
        self.conv3 = nn.Conv1d(16, 16, 5, stride=2)
        self.fc = nn.Linear(6*16+2+1+2+1, 64)

    def forward(self, obs, state=None, info={}):
        scan = th.as_tensor(obs.scan, device=self.device, dtype=th.float32)
        position = th.as_tensor(obs.position, device=self.device, dtype=th.float32)
        heading = th.as_tensor(obs.heading, device=self.device, dtype=th.float32)
        goal = th.as_tensor(obs.goal, device=self.device, dtype=th.float32)
        lact = th.as_tensor(obs.laction, device=self.device, dtype=th.float32)

        x = F.relu(self.conv1(scan))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.shape[0], -1)
        x = th.cat((x, position, heading, goal, lact), dim=-1)
        x = F.relu(self.fc(x))

        return x, state
