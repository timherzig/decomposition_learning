import os
import torch

import wandb
from torchvision.transforms import ToPILImage
from torch.nn import MSELoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from einops import rearrange
from models.up_scaling.unet.up_scale import UpSampler
from models.transformer.swin_transformer import SwinTransformer3D
import lightning.pytorch as pl


class Decomposer(pl.LightningModule):
    def __init__(self, config, log_dir: str = None):
        super().__init__()

        self.model_config = config.model
        self.train_config = config.train
        self.log_dir = log_dir

        self.validation_step_outputs = []
        self.best_val_loss = float("inf")

        if not self.model_config.swin.use_checkpoint:
            self.swin = SwinTransformer3D(patch_size=self.model_config.swin.patch_size)
        else:
            self.swin = SwinTransformer3D(
                pretrained=config.model.swin.checkpoint,
                patch_size=self.model_config.swin.patch_size,
            )
            self.swin.freeze()
            print(f"Loaded SWIN checkpoint")

        # Ground truth upsampling
        if self.model_config.upsampler_gt == "unet":
            self.decoder_gt_config = self.model_config.unet_gt.decoder

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
        if self.model_config.upsampler_sl == "unet":
            self.decoder_sl_config = self.model_config.unet_sl.decoder

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
        if self.model_config.upsampler_ob == "unet":
            self.decoder_ob_config = self.model_config.unet_ob.decoder

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

    def forward(self, x):
        x, encoder_features = self.swin(x)

        # Apply Upscaler_1 for reconstruction -> (B, 3, H, W)

        # Apply Upscaler_2 for shadow mask, light mask -> (B, 10, 2, H, W)

        # Apply Upscaler_3 for occlusion mask, occlusion rgb -> (B, 10, 4, H, W)

        # Perform upsampling if needed
        if self.model_config.upsampler_gt == "unet":
            gt_reconstruction = torch.squeeze(self.up_scale_gt(encoder_features[1:], x))
            if self.train_config.pre_train:
                return torch.clip(gt_reconstruction, -1.0, 1.0)

        if self.model_config.upsampler_sl == "unet":
            light_mask = self.up_scale_sl(encoder_features[1:], x)[:, 0, :, :, :]
            shadow_mask = self.up_scale_sl(encoder_features[1:], x)[:, 1, :, :, :]

        if self.model_config.upsampler_ob == "unet":
            occlusion_mask = self.up_scale_ob(encoder_features[1:], x)[:, 0, :, :, :]
            occlusion_rgb = self.up_scale_ob(encoder_features[1:], x)[:, 1:, :, :, :]

        return (
            torch.clip(gt_reconstruction, -1.0, 1.0),
            torch.clip(light_mask, -1.0, 1.0),
            torch.clip(shadow_mask, -1.0, 1.0),
            torch.clip(occlusion_mask, -1.0, 1.0),
            torch.clip(occlusion_rgb, -1.0, 1.0),
        )

    def weight_decay(self):
        decay = 0.0
        for name, param in self.named_parameters():
            if name in ["weights"]:
                decay += torch.sum(param**2)
        return decay * self.train_config.weight_decay

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

        reconstruction_loss = loss(reconstruction, input)

        return gt_loss + reconstruction_loss + self.weight_decay()

    def pre_train_loss(self, gt_reconstruction, input):
        loss = MSELoss()
        gt_loss = loss(gt_reconstruction, input)
        return gt_loss + self.weight_decay()

    def training_step(self, batch, batch_idx):
        (
            x,
            y,
        ) = batch  # --- x: (B, N, C, H, W), y: (B, C, H, W) | N: number of images in sequence

        if not self.train_config.pre_train:
            (
                gt_reconstruction,
                light_mask,
                shadow_mask,
                occlusion_mask,
                occlusion_rgb,
            ) = self(x)
        else:
            gt_reconstruction = self(x)

        loss = (
            self.loss_func(
                gt_reconstruction,
                light_mask,
                shadow_mask,
                occlusion_mask,
                occlusion_rgb,
                y,
                x,
            )
            if not self.train_config.pre_train
            else self.pre_train_loss(gt_reconstruction, x)
        )

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        (
            x,
            y,
        ) = batch  # --- x: (B, N, C, H, W), y: (B, C, H, W) | N: number of images in sequence

        if not self.train_config.pre_train:
            (
                gt_reconstruction,
                light_mask,
                shadow_mask,
                occlusion_mask,
                occlusion_rgb,
            ) = self(x)
        else:
            gt_reconstruction = self(x)

        loss = (
            self.loss_func(
                gt_reconstruction,
                light_mask,
                shadow_mask,
                occlusion_mask,
                occlusion_rgb,
                y,
                x,
            )
            if not self.train_config.pre_train
            else self.pre_train_loss(gt_reconstruction, x)
        )

        self.log("val_loss", loss, prog_bar=True, sync_dist=True)

        # Log images on the first validation step
        if batch_idx == 0 and not self.train_config.debug:
            self.log_images(
                y,
                x,
                gt_reconstruction,
                light_mask,
                shadow_mask,
                occlusion_mask,
                occlusion_rgb,
            ) if not self.train_config.pre_train else self.pre_train_log_images(
                gt_reconstruction, x
            )

        self.validation_step_outputs.append(loss)
        return loss

    def test_step(self, batch, batch_idx):
        (
            x,
            y,
        ) = batch  # --- x: (B, N, C, H, W), y: (B, C, H, W) | N: number of images in sequence

        if not self.train_config.pre_train:
            (
                gt_reconstruction,
                light_mask,
                shadow_mask,
                occlusion_mask,
                occlusion_rgb,
            ) = self(x)
        else:
            gt_reconstruction = self(x)

        loss = (
            self.loss_func(
                gt_reconstruction,
                light_mask,
                shadow_mask,
                occlusion_mask,
                occlusion_rgb,
                y,
                x,
            )
            if not self.train_config.pre_train
            else self.pre_train_loss(gt_reconstruction, x)
        )

        self.log("train_loss", loss, prog_bar=True)

        # Log images on the first test step
        if batch_idx == 0 and not self.train_config.debug:
            self.log_images(
                y,
                x,
                gt_reconstruction,
                light_mask,
                shadow_mask,
                occlusion_mask,
                occlusion_rgb,
            ) if not self.train_config.pre_train else self.pre_train_log_images(
                gt_reconstruction, x
            )
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=0.001)
        scheduler = ReduceLROnPlateau(optimizer, patience=10)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }

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

    def pre_train_log_images(self, gt_reconstruction, x):
        idx = torch.randint(0, x.shape[0], (1,)).item()

        x = x[idx, :, :, :, :]
        gt_reconstruction = gt_reconstruction[idx, :, :, :]

        columns = ["input", "gt_reconstruction"]

        my_data = [
            [
                [
                    wandb.Image(self.to_pil(x[:, img, :, :]), caption=columns[0])
                    for img in range(x.shape[1])
                ],
                [
                    wandb.Image(
                        self.to_pil(gt_reconstruction[:, img, :, :]), caption=columns[1]
                    )
                    for img in range(gt_reconstruction.shape[1])
                ],
            ]
        ]

        self.logger.log_table(key="input_output", columns=columns, data=my_data)

    def on_validation_epoch_end(self) -> None:
        loss = torch.stack(self.validation_step_outputs).mean()
        if loss < self.best_val_loss and self.train_config.pre_train:
            self.best_val_loss = loss

            # Save only the swin part of the encoder
            torch.save(
                self.swin.state_dict(),
                f"{self.log_dir}/swin_encoder.pt",
            )
        self.validation_step_outputs.clear()
