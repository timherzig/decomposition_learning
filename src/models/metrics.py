from typing import Any
import torch
import torch.nn as nn
from torch.nn import MSELoss, L1Loss
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
        self.metric = SSIMLoss(data_range=1.0)

    def __call__(self, x, y):
        # self.metric.to(x.device)
        loss = self.metric(x, y)
        print(loss)
        return 1.0 - loss


class MAE:
    def __init__(self):
        super().__init__()
        self.metric = L1Loss()

    def __call__(self, x, y):
        self.metric.to(x.device)
        return self.metric(x, y)
