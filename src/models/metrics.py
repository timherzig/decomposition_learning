import torch
import torch.nn as nn
from torch.nn import MSELoss
from torchmetrics import StructuralSimilarityIndexMeasure as SSIMLoss


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(x, y):
        return MSELoss(x, y)


class SSIMLoss(nn.Module):
    def __init__(self):
        super(SSIMLoss, self).__init__()

    def forward(x, y):
        return SSIMLoss(x, y)
