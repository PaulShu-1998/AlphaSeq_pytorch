import torch.nn as nn
import torch.nn.functional as F
from const import BLOCKS


class BasicBlock(nn.Module):
    """Basic residual block with 2 convolutions and a skip connection before the last activation."""
    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.conv2(out)
        out = F.relu(self.bn2(out))
        out += residual
        out = F.relu(out)
        return out


class Extractor(nn.Module):
    """
    This net work is used as a feature extractor, and takes as input the 'state' defined as (N,M).
    """
    def __init__(self, inplanes, outplanes):
        super(Extractor, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, outplanes, stride=1, kernel_size=3, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(outplanes)

        for block in range(BLOCKS):
            setattr(self, "res{}".format(block), BasicBlock(outplanes, outplanes))

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        for block in range(BLOCKS-1):
            x = getattr(self, "res{}".format(block))(x)

        feature_map = getattr(self, "res{}".format(BLOCKS-1))(x)
        return feature_map