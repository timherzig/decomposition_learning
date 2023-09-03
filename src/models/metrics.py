from typing import Any
import torch
import torch.nn as nn
from torch.nn import MSELoss, L1Loss, BCELoss, BCEWithLogitsLoss
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIMLoss


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
        self.metric.to(x.device)
        loss = self.metric(x, y)
        return 1.0 - loss


class MAE:
    def __init__(self):
        super().__init__()
        self.metric = L1Loss()

    def __call__(self, x, y):
        self.metric.to(x.device)
        return self.metric(x, y)


class MAE_weighted:
    def __init__(self):
        super().__init__()
        self.weight_factor = 50.0

    def __call__(self, x, y):
        # weight is whever there is a non zero value in y (i.e. the ground truth)
        weight_matrix = torch.ones_like(y) * self.weight_factor
        weight = torch.where(y != 0, weight_matrix, torch.ones_like(y))
        loss = torch.mean(torch.abs(x - y) * weight)
        return loss


class BCE:
    def __init__(self):
        super().__init__()
        self.metric = BCELoss()

    def __call__(self, x, y):
        self.metric.to(x.device)
        return self.metric(x, y)


class BCEWithLogits:
    def __init__(self, pos_weight=torch.tensor([1.0])):
        super().__init__()
        self.metric = BCEWithLogitsLoss(pos_weight=pos_weight)

    def __call__(self, x, y):
        self.metric.to(x.device)
        return self.metric(x, y)
