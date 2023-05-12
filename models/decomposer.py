import os
import torch

from torch.nn import MSELoss
from einops import rearrange
from models.up_scaling.unet.up_scale import UpSampler
from models.transformer.swin_transformer import SwinTransformer3D


class Decomposer(SwinTransformer3D): 
    def __init__(self, config):
        super().__init__(patch_size = config.swin.patch_size)

        self.config = config
        if(config.upsampler == "unet"):
            self.decoder_config = config.unet.decoder

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
        if(self.config.upsampler is not None):
            x = self.up_scale(encoder_features[1:], x)

        return x
    
    def loss_func(self, prediction, target):
        prediction = torch.mean(prediction, 2) # --- average of N predictions
        loss = MSELoss()
        return loss(prediction, target)

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = self.loss_func(output, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = self.loss_func(output, y)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02) 
    


