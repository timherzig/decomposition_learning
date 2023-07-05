import torch
import torch.nn as nn
from torch.nn import MSELoss
from torchmetrics import StructuralSimilarityIndexMeasure as SSIMLoss


class MSE:
    def __init__(self):
        super().__init__()
        self.metric = MSELoss()

    def __call__(self, x, y):
        return self.metric(x, y)


class SSIM:
    def __init__(self):
        super().__init__()
        self.metric = SSIMLoss()

    def __call__(self, x, y):
        self.metric.to(x.device)
        return self.metric(x, y)
