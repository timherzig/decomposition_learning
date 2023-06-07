import os
import torch
import cv2

import wandb
from torchvision.transforms import ToPILImage
from torch.nn import MSELoss
from einops import rearrange
from models.up_scaling.unet.up_scale import UpSampler
from models.transformer.swin_transformer import SwinTransformer3D
from models.up_scaling.reverse_st.upsampling import SwinTransformer3D_up


class Decomposer(SwinTransformer3D):
    def __init__(self, config):
        super().__init__(patch_size=config.swin.patch_size)

        self.config = config

        # Ground truth upsampling
        if config.upsampler_gt == "unet":
            self.decoder_gt_config = config.unet_gt.decoder

            self.up_scale_gt = UpSampler(
                self.decoder_gt_config.f_maps,
                self.decoder_gt_config.conv_kernel_size,
                self.decoder_gt_config.conv_padding,
                self.decoder_gt_config.layer_order,
                self.decoder_gt_config.num_groups,
                self.decoder_gt_config.is3d,
                self.decoder_gt_config.output_dim,
                self.decoder_gt_config.layers_no_skip.scale_factor,
                self.decoder_gt_config.layers_no_skip.size,
            )

        # Shadow and light upsampling
        if config.upsampler_sl == "unet":
            self.decoder_sl_config = config.unet_sl.decoder

            self.up_scale_sl = UpSampler(
                self.decoder_sl_config.f_maps,
                self.decoder_sl_config.conv_kernel_size,
                self.decoder_sl_config.conv_padding,
                self.decoder_sl_config.layer_order,
                self.decoder_sl_config.num_groups,
                self.decoder_sl_config.is3d,
                self.decoder_sl_config.output_dim,
                self.decoder_sl_config.layers_no_skip.scale_factor,
                self.decoder_sl_config.layers_no_skip.size,
            )

        # Object upsampling
        if config.upsampler_ob == "unet":
            self.decoder_ob_config = config.unet_ob.decoder

            self.up_scale_ob = UpSampler(
                self.decoder_ob_config.f_maps,
                self.decoder_ob_config.conv_kernel_size,
                self.decoder_ob_config.conv_padding,
                self.decoder_ob_config.layer_order,
                self.decoder_ob_config.num_groups,
                self.decoder_ob_config.is3d,
                self.decoder_ob_config.output_dim,
                self.decoder_ob_config.layers_no_skip.scale_factor,
                self.decoder_ob_config.layers_no_skip.size,
            )

        self.to_pil = ToPILImage()

    # Override original swin forward function
    def forward(self, x):
        if self.config.upsampler == "swin": 
            x = self.patch_embed(x)
            x = self.pos_drop(x)

            for idx, layer in enumerate(self.layers):
                x, _ = layer(x.contiguous())
            x = rearrange(x, "n c d h w -> n d h w c")
            x = self.norm(x)
            x = rearrange(x, "n d h w c -> n c d h w")

            # Perform upsampling 
            model_gt = SwinTransformer3D_up()
            gt_reconstruction = model_gt.forward(x, 3) 
            gt_reconstruction = torch.squeeze(gt_reconstruction) 
            model_light = SwinTransformer3D_up()
            light_mask = model_light.forward(x, 1)
            model_shadow = SwinTransformer3D_up()
            shadow_mask = model_shadow.forward(x, 1)
            model_ob = SwinTransformer3D_up()
            occlusion = model_ob.forward(x, 4)
            occlusion_mask = occlusion[:, 0, :, :, :]
            occlusion_rgb = occlusion[:, 1:, :, :, :]

        else: 
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
            if self.config.upsampler_gt == "unet":
                gt_reconstruction = torch.squeeze(self.up_scale_gt(encoder_features[1:], x))

            if self.config.upsampler_sl == "unet":
                light_mask = self.up_scale_sl(encoder_features[1:], x)[:, 0, :, :, :]
                shadow_mask = self.up_scale_sl(encoder_features[1:], x)[:, 1, :, :, :]

            if self.config.upsampler_ob == "unet":
                occlusion_mask = self.up_scale_ob(encoder_features[1:], x)[:, 0, :, :, :]
                occlusion_rgb = self.up_scale_ob(encoder_features[1:], x)[:, 1:, :, :, :]

        return gt_reconstruction, light_mask, shadow_mask, occlusion_mask, occlusion_rgb

    def loss_func(
        self,
        gt_reconstruction,
        light_mask,
        shadow_mask,
        occlusion_mask,
        occlusion_rgb,
        target,
        input,
    ):
        loss = MSELoss()
        gt_loss = loss(gt_reconstruction, target)

        gt_reconstruction = gt_reconstruction.unsqueeze(2).repeat(1, 1, 10, 1, 1)
        shadow_mask = shadow_mask.unsqueeze(1).repeat(1, 3, 1, 1, 1)
        light_mask = light_mask.unsqueeze(1).repeat(1, 3, 1, 1, 1)
        occlusion_mask = occlusion_mask.unsqueeze(1).repeat(1, 3, 1, 1, 1)

        reconstruction = torch.where(
            occlusion_mask < 0.5,
            (gt_reconstruction * shadow_mask + light_mask),
            occlusion_rgb,
        )

        # reconstruction = torch.where(
        #     occlusion_mask < 0.5,
        #     (gt_reconstruction * shadow_mask + light_mask),
        #     (gt_reconstruction * shadow_mask + light_mask) + occlusion_rgb,
        # )
        # another form
        #reconstruction = cv2.bitwise_and((gt_reconstruction * shadow_mask + light_mask), occlusion_mask),

        reconstruction_loss = loss(reconstruction, input)

        return gt_loss + reconstruction_loss

    def training_step(self, batch, batch_idx):
        (
            x,
            y,
        ) = batch  # --- x: (B, N, C, H, W), y: (B, C, H, W) | N: number of images in sequence

        (
            gt_reconstruction,
            light_mask,
            shadow_mask,
            occlusion_mask,
            occlusion_rgb,
        ) = self(x)

        loss = self.loss_func(
            gt_reconstruction,
            light_mask,
            shadow_mask,
            occlusion_mask,
            occlusion_rgb,
            y,
            x,
        )

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        (
            x,
            y,
        ) = batch  # --- x: (B, N, C, H, W), y: (B, C, H, W) | N: number of images in sequence

        (
            gt_reconstruction,
            light_mask,
            shadow_mask,
            occlusion_mask,
            occlusion_rgb,
        ) = self(x)

        loss = self.loss_func(
            gt_reconstruction,
            light_mask,
            shadow_mask,
            occlusion_mask,
            occlusion_rgb,
            y,
            x,
        )

        self.log("val_loss", loss, prog_bar=True)

        # Log images on the first validation step
        if batch_idx == 0:
            self.log_images(
                y,
                x,
                gt_reconstruction,
                light_mask,
                shadow_mask,
                occlusion_mask,
                occlusion_rgb,
            )
        return loss

    def test_step(self, batch, batch_idx):
        (
            x,
            y,
        ) = batch  # --- x: (B, N, C, H, W), y: (B, C, H, W) | N: number of images in sequence

        (
            gt_reconstruction,
            light_mask,
            shadow_mask,
            occlusion_mask,
            occlusion_rgb,
        ) = self(x)

        loss = self.loss_func(
            gt_reconstruction,
            light_mask,
            shadow_mask,
            occlusion_mask,
            occlusion_rgb,
            y,
            x,
        )

        self.log("train_loss", loss, prog_bar=True)

        # Log images on the first test step
        if batch_idx == 0:
            self.log_images(
                y,
                x,
                gt_reconstruction,
                light_mask,
                shadow_mask,
                occlusion_mask,
                occlusion_rgb,
            )
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def log_images(
        self,
        y,
        x,
        gt_reconstruction,
        shadow_mask,
        light_mask,
        occlusion_mask,
        occlusion_rgb,
    ):
        """
        Logs one image sequence to the wandb logger

        params:
            y: unoccluded original input (1)
            x: occluded input (10)
            gt_reconstruction: ground truth reconstruction (1)
            shadow_mask: shadow mask (10)
            light_mask: light mask (10)
            occlusion_mask: occlusion mask (10)
            occlusion_rgb: occlusion rgb (10)
        """
        idx = torch.randint(0, y.shape[0], (1,)).item()

        y = y[idx, :, :, :]
        x = x[idx, :, :, :, :]
        gt_reconstruction = gt_reconstruction[idx, :, :, :]
        shadow_mask = shadow_mask[idx, :, :, :]
        light_mask = light_mask[idx, :, :, :]
        occlusion_mask = occlusion_mask[idx, :, :, :]
        occlusion_rgb = occlusion_rgb[idx, :, :, :, :]

        columns = [
            "org_img",
            "org_occlusion",
            "gt_reconstruction",
            "shadow_mask",
            "light_mask",
            "occlusion_mask",
            "occlusion_rgb",
            "occlusion_reconstruction",
        ]

        occlusion_rec = torch.where(
            occlusion_mask.unsqueeze(0).repeat(3, 1, 1, 1) < 0.5,
            (
                gt_reconstruction.unsqueeze(1).repeat(1, 10, 1, 1)
                * shadow_mask.unsqueeze(0).repeat(3, 1, 1, 1)
                + light_mask.unsqueeze(0).repeat(3, 1, 1, 1)
            ),
            occlusion_rgb,
        )

        my_data = [
            [
                wandb.Image(self.to_pil(y), caption=columns[0]),
                [
                    wandb.Image(self.to_pil(x[:, img, :, :]), caption=columns[1])
                    for img in range(x.shape[1])
                ],
                wandb.Image(self.to_pil(gt_reconstruction), caption=columns[2]),
                [
                    wandb.Image(self.to_pil(shadow_mask[img, :, :]), caption=columns[3])
                    for img in range(shadow_mask.shape[0])
                ],
                [
                    wandb.Image(self.to_pil(light_mask[img, :, :]), caption=columns[4])
                    for img in range(light_mask.shape[0])
                ],
                [
                    wandb.Image(
                        self.to_pil(occlusion_mask[img, :, :]), caption=columns[5]
                    )
                    for img in range(occlusion_mask.shape[0])
                ],
                [
                    wandb.Image(
                        self.to_pil(occlusion_rgb[:, img, :, :]), caption=columns[6]
                    )
                    for img in range(occlusion_rgb.shape[1])
                ],
                [
                    wandb.Image(
                        self.to_pil(occlusion_rec[:, img, :, :]), caption=columns[7]
                    )
                    for img in range(occlusion_rec.shape[1])
                ],
            ]
        ]

        self.logger.log_table(key="input_output", columns=columns, data=my_data)
