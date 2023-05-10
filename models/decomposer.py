import os
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import lightning.pytorch as pl
import torch
from models.transformer.swin_transformer import SwinTransformer3D
from timm.models.layers import DropPath, trunc_normal_
from einops import rearrange
from models.up_scaling.unet.unet3d import create_decoders


class Decomposer(SwinTransformer3D): 
    def __init__(self, unet_config):
        super().__init__()
        self.unet_config = unet_config

        # self.encoder = create_decoders(unet_config.f_maps, unet_config.basic_module, unet_config.conv_kernel_size, unet_config.conv_padding, unet_config.layer_order, unet_config.num_groups,
        #                                 unet_config.is3d)

    # Override original swin forward function
    def forward(self, x):
        x = self.patch_embed(x)
        print("after patch parition: ", x.shape)
        x = self.pos_drop(x)

        for idx, layer in enumerate(self.layers):
            x = layer(x.contiguous())
            print("Layer nr:" ,str(idx), " shape: ", x.shape)

        x = rearrange(x, 'n c d h w -> n d h w c')
        x = self.norm(x)
        x = rearrange(x, 'n d h w c -> n c d h w')

        
        return x
    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.type(torch.FloatTensor)
        print("Input shape: ", x.shape)
        output = self(x)
        print("output shape: ", output.shape)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02) 
    


