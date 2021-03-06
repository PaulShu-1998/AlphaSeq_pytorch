import torch.nn as nn
import torch.nn.functional as F
from torch import tanh
from const import N, M


class ValueNet(nn.Module):
    """
    This network is used to predict which player is more likely to win given the input 'state'
    described in the Feature Extractor model.
    The output is a continuous variable, between -1 and 1.
    """

    def __init__(self, inplanes):
        super(ValueNet, self).__init__()
        self.conv = nn.Conv2d(inplanes, 1, kernel_size=1)
        self.bn = nn.BatchNorm2d(1)
        self.fc1 = nn.Linear(1*N*M, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        """
        x : feature maps extracted from the state
        winning : probability of the current agent winning the game
                  considering the actual state of the board
        """

        x = F.relu(self.bn(self.conv(x)))
        x = x.view(-1, N*M)
        x = F.relu(self.fc1(x))
        reward = tanh(self.fc2(x))

        return reward