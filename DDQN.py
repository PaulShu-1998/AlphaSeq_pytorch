import gym
import torch
from collections import deque
import random
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class DuelingModel(nn.Module):
    def __init__(self, n_input, n_output, n_hidden):
        super(DuelingModel, self).__init__()
        self.adv1 = nn.Linear(n_input, n_hidden)
        self.adv2 = nn.Linear(n_hidden, n_output)
        self.val1 = nn.Linear(n_input, n_hidden)
        self.val2 = nn.Linear(n_hidden, 1)

    def forward(self, x):
        adv = nn.functional.relu(self.adv1(x))
        adv = self.adv2(adv)
        val = nn.functional.relu(self.val1(x))
        val = self.val2(val)
        return val + adv - adv.mean()

class CNNModel(nn.Module):
    def __init__(self, n_channel, n_action):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=n_channel, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv2 = nn.Conv2d(64, 64, 3, stride=1)
        self.fc = nn.Linear(7 * 7 * 64, 512)
        self.out = nn.Linear(512, n_action)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        output = self.out(x)
        return output



class DQN():
    def __init__(self, n_state, n_action, n_hidden=50, lr=0.05):
        self.criterion = nn.MSELoss()
        self.model = DuelingModel(n_state, n_action, n_hidden)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)