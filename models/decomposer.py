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
from models.up_scaling.unet.up_scale import UpSampler


class Decomposer(SwinTransformer3D): 
    def __init__(self, swin_config, up_sampling = None, unet_config = None):
        super().__init__(patch_size = swin_config.patch_size)
        self.up_sampling = up_sampling
        if(up_sampling == "unet"):
            self.decoder_config = unet_config.decoder

            self.up_scale = UpSampler(self.decoder_config.f_maps,
                                      self.decoder_config.conv_kernel_size,
                                      self.decoder_config.conv_padding,
                                      self.decoder_config.layer_order,
                                      self.decoder_config.num_groups,
                                      self.decoder_config.is3d,
                                      self.decoder_config.output_dim,
                                      self.decoder_config.layers_no_skip.scale_factor, 
                                      self.decoder_config.layers_no_skip.size
                                      )
            

    # Override original swin forward function
    def forward(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        # collect layers from encoder part 
        encoder_features = []
        for idx, layer in enumerate(self.layers):
            x, x_no_merge = layer(x.contiguous())
            encoder_features.insert(0, x_no_merge)
        x = rearrange(x, 'n c d h w -> n d h w c')
        x = self.norm(x)
        x = rearrange(x, 'n d h w c -> n c d h w')

        # Perform upsampling if needed
        if(self.up_sampling is not None):
            x = self.up_scale(encoder_features[1:], x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.type(torch.FloatTensor)
        output = self(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02) 
    


