import os
import torch

import wandb
from torchvision.transforms import ToPILImage
from torch.nn import MSELoss
from einops import rearrange
from models.up_scaling.unet.up_scale import UpSampler
from models.transformer.swin_transformer import SwinTransformer3D


class Decomposer(SwinTransformer3D):
    def __init__(self, config):
        super().__init__(patch_size=config.swin.patch_size)

        self.config = config
        if config.upsampler == "unet":
            self.decoder_config = config.unet.decoder

            self.up_scale = UpSampler(
                self.decoder_config.f_maps,
                self.decoder_config.conv_kernel_size,
                self.decoder_config.conv_padding,
                self.decoder_config.layer_order,
                self.decoder_config.num_groups,
                self.decoder_config.is3d,
                self.decoder_config.output_dim,
                self.decoder_config.layers_no_skip.scale_factor,
                self.decoder_config.layers_no_skip.size,
            )

        self.to_pil = ToPILImage()

    # Override original swin forward function
    def forward(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        # collect layers from encoder part
        encoder_features = []
        for idx, layer in enumerate(self.layers):
            x, x_no_merge = layer(x.contiguous())
            encoder_features.insert(0, x_no_merge)
        x = rearrange(x, "n c d h w -> n d h w c")
        x = self.norm(x)
        x = rearrange(x, "n d h w c -> n c d h w")

        # Perform upsampling if needed
        if self.config.upsampler == "unet":
            x = self.up_scale(encoder_features[1:], x)

        return x

    def loss_func(self, prediction, target, input):
        gt_reconstruction = torch.mean(
            prediction[:, :3, :, :, :], 2
        )  # --- average of N predictions
        light_mask = prediction[:, 3, :, :, :].unsqueeze(1)
        shadow_mask = prediction[:, 4, :, :, :].unsqueeze(1)
        occlusion_mask = prediction[:, 5, :, :, :].unsqueeze(1)
        occlusion_rgb = prediction[:, 6:, :, :, :]

        loss = MSELoss()
        gt_loss = loss(gt_reconstruction, target)

        gt_reconstruction = gt_reconstruction.unsqueeze(2).repeat(1, 1, 10, 1, 1)
        reconstruction = torch.where(
            occlusion_mask < 0.5,
            (gt_reconstruction * shadow_mask + light_mask),
            occlusion_rgb,
        )

        reconstruction_loss = loss(reconstruction, input)

        return gt_loss + reconstruction_loss

    def training_step(self, batch, batch_idx):
        (
            x,
            y,
        ) = batch  # --- x: (B, N, C, H, W), y: (B, C, H, W) | N: number of images in sequence
        output = self(x)  # --- output: (B, C, N, H, W)
        loss = self.loss_func(output, y, x)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        (
            x,
            y,
        ) = batch  # --- x: (B, N, C, H, W), y: (B, C, H, W) | N: number of images in sequence
        output = self(x)  # --- output: (B, C, N, H, W)
        loss = self.loss_func(output, y, x)
        self.log("val_loss", loss, prog_bar=True)

        # Log images on the first validation step
        if batch_idx == 0:
            self.log_images(x, output, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

    def log_images(self, x, output, y):
        """log one random input output pair from batch as table
        Columns: input, output, merged output, target

            Args:
                x (torch.Tensor): input tensor. Shape: (B, N, C, H, W) \\
                output (torch.Tensor): output tensor. Shape: (B, C, N, H, W) \\
                y (torch.Tensor): target tensor. Shape: (B, C, H, W)
        """
        output = output[:, :3, :, :, :]
        idx = torch.randint(0, x.shape[0], (1,)).item()
        columns = ["input", "output", "merged output", "target"]
        my_data = [
            [
                [
                    wandb.Image(self.to_pil(x[idx, :, img, :, :]), caption=columns[0])
                    for img in range(x.shape[1])
                ],
                [
                    wandb.Image(
                        self.to_pil(output[idx, :, img, :, :]),
                        caption=columns[1],
                    )
                    for img in range(output.shape[1])
                ],
                wandb.Image(
                    self.to_pil(torch.mean(output[idx], 1)),
                    caption=columns[2],
                ),
                wandb.Image(self.to_pil(y[idx, :, :, :]), caption=columns[3]),
            ]
        ]
        self.logger.log_table(key="input_output", columns=columns, data=my_data)
