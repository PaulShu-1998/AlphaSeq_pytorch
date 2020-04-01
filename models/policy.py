import torch.nn as nn
import torch.nn.functional as F
from const import N, M, q


class PolicyNet(nn.Module):
    """
    This network is used in order to predict which move has the best potential to lead to a win
    given the same 'state' described in the Feature Extractor model.
    """

    def __init__(self, inplanes):
        super(PolicyNet, self).__init__()
        self.conv = nn.Conv2d(inplanes, 1, kernel_size=1)
        self.bn = nn.BatchNorm2d(1)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.fc = nn.Linear(N*M, q**N)

    def forward(self, x):
        """
        x : feature maps extracted from the state
        probas : a 1*N vector where N is the length of the sequence
                 Each value in this vector represent the likelihood
                 of next action
        """

        x = F.relu(self.bn(self.conv(x)))
        x = x.view(-1, N*M)
        x = self.fc(x)
        probs = self.logsoftmax(x).exp()

        return probs