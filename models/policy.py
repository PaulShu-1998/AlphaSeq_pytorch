import torch.nn as nn
import torch.nn.functional as F
from const import N, M, OUTPLANES


class PolicyNet(nn.Module):
    """
    This network is used in order to predict which move has the best potential to lead to a win
    given the same 'state' described in the Feature Extractor model.
    """

    def __init__(self, inplanes):
        super(PolicyNet, self).__init__()
        self.conv = nn.Conv2d(inplanes, 2, kernel_size=1)
        self.bn = nn.BatchNorm2d(2)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.fc1 = nn.Linear(2*N*M, 512)
        self.fc2 = nn.Linear(512, OUTPLANES)

    def forward(self, x):
        """
        x : feature maps extracted from the state
        probas : a 1*N vector where N is the length of the sequence
                 Each value in this vector represent the likelihood
                 of next action
        """

        x = F.relu(self.bn(self.conv(x)))
        x = x.view(-1, 2*N*M)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        probs = self.logsoftmax(x).exp()

        return probs