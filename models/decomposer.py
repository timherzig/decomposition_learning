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
    def __init__(self, unet_config):
        super().__init__(patch_size=(2, 4, 4))
        self.decoder_config = unet_config.decoder

        self.encoder = create_decoders(self.decoder_config.f_maps, DoubleConv, self.decoder_config.conv_kernel_size, self.decoder_config.conv_padding, self.decoder_config.layer_order, self.decoder_config.num_groups,
                                        self.decoder_config.is3d)

    # Override original swin forward function
    def forward(self, x):
        x = self.patch_embed(x)
        print("after patch parition: ", x.shape)
        x = self.pos_drop(x)

        encoder_layers = []
        for idx, layer in enumerate(self.layers):
            x, x_no_merge = layer(x.contiguous())
            # if idx < (self.num_layers -1):
            #     encoder_layers.append(x)
            print("Layer nr:" ,str(idx), " shape: ", x_no_merge.shape)

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
    


