import os
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import lightning.pytorch as pl
import torch
from swin_transformer import SwinTransformer3D
from timm.models.layers import DropPath, trunc_normal_
from einops import rearrange

class Decomposer(SwinTransformer3D): 
    def __init__(self):
        super().__init__()

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.type(torch.FloatTensor)
        print("Input shape: ", x.shape)
        output = self(x)
        print("output shape: ", output.shape)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02) 
    


