import torch
import torch.nn as nn
from torch.nn import MSELoss
from torchmetrics import StructuralSimilarityIndexMeasure as SSIM


class MSE:
    def __init__(self):
        super().__init__()
        self.metric = MSELoss()

    def __call__(self, x, y):
        return self.metric(x, y)


class SSIM:
    def __init__(self):
        super().__init__()
        self.metric = SSIM()

    def __call__(self, x, y):
        return self.metric(x, y)
