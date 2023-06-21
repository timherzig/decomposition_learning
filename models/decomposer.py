import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import lightning.pytorch as pl

from models.transformer.swin_transformer import SwinTransformer3D
from models.up_scaling.unet.up_scale import UpSampler
from models.util.loss_functions import (
    base_loss,
    reconstruction_loss,
    regularized_loss,
    pre_train_loss,
)
from utils.wandb_logging import (
    log_images,
    pre_train_log_images,
)


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
                frozen_stages=-1,  # =0
            )
            print(f"Loaded SWIN checkpoint")
            print("-----------------")

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
                self.decoder_gt_config.omit_skip_connections,
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
                self.decoder_sl_config.omit_skip_connections,
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
                self.decoder_ob_config.omit_skip_connections,
            )

    def forward(self, x):
        x, encoder_features = self.swin(x)
        if self.train_config.debug:
            print(f"swin x shape: {x.shape}")
            for idx, ef in enumerate(encoder_features):
                print(f"swin encoder_feature {idx} shape: {ef.shape}")

        # Apply Upscaler_1 for reconstruction -> (B, 3, H, W)
        if self.model_config.upsampler_gt == "unet":
            gt_reconstruction = torch.squeeze(self.up_scale_gt(encoder_features[1:], x))
            if self.train_config.pre_train:
                return torch.clip(gt_reconstruction, -1.0, 1.0)

        # Apply Upscaler_2 for shadow mask, light mask -> (B, 10, 2, H, W)
        if self.model_config.upsampler_sl == "unet":
            light_mask = self.up_scale_sl(encoder_features[1:], x)[:, 0, :, :, :]
            shadow_mask = self.up_scale_sl(encoder_features[1:], x)[:, 1, :, :, :]

        # Apply Upscaler_3 for occlusion mask, occlusion rgb -> (B, 10, 4, H, W)
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
        if self.train_config.pre_train:
            return pre_train_loss(
                gt_reconstruction, input, self, self.train_config.weight_decay
            )

        if self.model_config.checkpoint:
            return reconstruction_loss(
                gt_reconstruction,
                light_mask,
                shadow_mask,
                occlusion_mask,
                occlusion_rgb,
                target,
                input,
                self,
                self.train_config.weight_decay,
            )

        if self.train_config.loss_func == "base_loss":
            return base_loss(
                gt_reconstruction,
                light_mask,
                shadow_mask,
                occlusion_mask,
                occlusion_rgb,
                target,
                input,
                self,
                self.train_config.weight_decay,
            )
        elif self.train_config.loss_func == "regularized_loss":
            return regularized_loss(
                gt_reconstruction,
                light_mask,
                shadow_mask,
                occlusion_mask,
                occlusion_rgb,
                target,
                input,
                self,
                self.train_config.weight_decay,
                self.train_config.mask_decay,
                self.train_config.lambda_gt_loss,
                self.train_config.lambda_decomp_loss,
                self.train_config.lambda_occlusion_difference,
            )
        else:
            raise NotImplementedError

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

        loss = self.loss_func(
            gt_reconstruction,
            light_mask,
            shadow_mask,
            occlusion_mask,
            occlusion_rgb,
            y,
            x,
        )

        self.log("val_loss", loss, prog_bar=True, sync_dist=True)

        # Log images on the first validation step
        if batch_idx == 0 and not self.train_config.debug:
            pre_train_log_images(
                gt_reconstruction, x
            ) if self.train_config.pre_train else log_images(
                y,
                x,
                gt_reconstruction,
                light_mask,
                shadow_mask,
                occlusion_mask,
                occlusion_rgb,
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
        if batch_idx == 0 and not self.train_config.debug:
            pre_train_log_images(
                gt_reconstruction, x
            ) if self.train_config.pre_train else log_images(
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
        optimizer = Adam(self.parameters(), lr=self.train_config.lr)
        scheduler = ReduceLROnPlateau(optimizer, patience=10)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }

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
