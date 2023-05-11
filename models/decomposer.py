import os
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import lightning.pytorch as pl
import torch
from models.transformer.swin_transformer import SwinTransformer3D
from timm.models.layers import DropPath, trunc_normal_
from einops import rearrange
from models.up_scaling.unet.unet3d import create_decoders, DoubleConv


class Decomposer(SwinTransformer3D): 
    def __init__(self, swin_config, up_sampling = None, unet_config = None):
        super().__init__(swin_config.patch_size)

        if(up_sampling == "unet"):
            self.decoder_config = unet_config.decoder

            self.encoder = create_decoders(self.decoder_config.f_maps, DoubleConv, self.decoder_config.conv_kernel_size, self.decoder_config.conv_padding, self.decoder_config.layer_order, self.decoder_config.num_groups,
                                            self.decoder_config.is3d)

    # Override original swin forward function
    def forward(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        for idx, layer in enumerate(self.layers):
            x = layer(x.contiguous())

        x = rearrange(x, 'n c d h w -> n d h w c')
        x = self.norm(x)
        x = rearrange(x, 'n d h w c -> n c d h w')

        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.type(torch.FloatTensor)
        output = self(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02) 
    


