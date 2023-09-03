import os

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import lightning.pytorch as pl
import torch.nn.functional as F

from src.models.transformer.swin_transformer import SwinTransformer3D
from src.models.up_scaling.unet.up_scale import UpSampler
from src.models.utils.utils import get_class

from utils.wandb_logging import (
    log_images,
    pre_train_log_images,
)

from utils.logging import evaluation_log_images
from einops import rearrange


class Decomposer(pl.LightningModule):
    def __init__(self, config, log_dir: str = None, eval_output: str = None):
        super().__init__()

        self.data_config = config.data
        self.model_config = config.model
        self.train_config = config.train
        self.log_dir = log_dir
        self.eval_output = eval_output

        self.validation_step_outputs = []
        self.best_val_loss = float("inf")

        # --- Loss ---
        loss_class = get_class(self.train_config.loss_func, ["src.models.losses"])
        self.loss = loss_class(self, self.train_config)
        # ------------

        # --- Enocder ---
        if not self.model_config.swin.use_checkpoint:
            self.swin = SwinTransformer3D(patch_size=self.model_config.swin.patch_size)
        else:
            self.swin = SwinTransformer3D(
                pretrained=f"{os.getcwd()}/{config.model.swin.checkpoint}",
                patch_size=self.model_config.swin.patch_size,
                frozen_stages=self.model_config.swin.frozen_stages,
            )
            # freeze SWIN
            for param in self.swin.parameters():
                param.requires_grad = False
            print("Loaded SWIN checkpoint")
            print("-----------------")
        # ----------------

        # --- Upscaler ---
        # Ground truth upsampling
        if self.model_config.upsampler_gt == "unet":
            self.decoder_gt_config = self.model_config.unet_gt.decoder
            arguments = dict(self.decoder_gt_config)
            self.up_scale_gt = UpSampler(**arguments)
            if self.model_config.unet_gt.checkpoint is not False:
                print("Loading UNet GT")
                self.up_scale_gt.load_state_dict(
                    torch.load(
                        f"{os.getcwd()}/{self.model_config.unet_gt.checkpoint}",
                        map_location=torch.device(self.train_config.device),
                    ),
                )
            if self.model_config.unet_gt.freeze:
                print("Freezing UNet GT")
                for param in self.up_scale_gt.parameters():
                    param.requires_grad = False

        # Shadow and light upsampling
        if self.model_config.upsampler_sl == "unet":
            self.decoder_sl_config = self.model_config.unet_sl.decoder
            arguments = dict(self.decoder_sl_config)
            self.up_scale_sl = UpSampler(**arguments)
            if self.model_config.unet_sl.checkpoint is not False:
                print("Loading UNet SL")
                self.up_scale_sl.load_state_dict(
                    torch.load(
                        f"{os.getcwd()}/{self.model_config.unet_sl.checkpoint}",
                        map_location=torch.device(self.train_config.device),
                    )
                )
            if self.model_config.unet_sl.freeze:
                print("Freezing UNet SL")
                for param in self.up_scale_sl.parameters():
                    param.requires_grad = False

        # Object upsampling
        if self.model_config.upsampler_ob == "unet":
            self.decoder_ob_config = self.model_config.unet_ob.decoder
            arguments = dict(self.decoder_ob_config)
            self.up_scale_ob = UpSampler(**arguments)
            if self.model_config.unet_ob.checkpoint is not False:
                print("Loading UNet OCC")
                self.up_scale_ob.load_state_dict(
                    torch.load(
                        f"{os.getcwd()}/{self.model_config.unet_ob.checkpoint}",
                        map_location=torch.device(self.train_config.device),
                    )
                )
            if self.model_config.unet_ob.freeze:
                print("Freezing UNet OCC")
                for param in self.up_scale_ob.parameters():
                    param.requires_grad = False

    def forward(self, x):
        return self.forward_unet(x)

    def forward_unet(self, x):
        x, encoder_features = self.swin(x)
        # Apply Upscaler_1 for reconstruction -> (B, 3, H, W)
        if self.model_config.upsampler_gt == "unet":
            gt_reconstruction = F.sigmoid(
                torch.squeeze(self.up_scale_gt(encoder_features[1:], x))
            )
            if self.train_config.pre_train:
                return torch.clip(gt_reconstruction, 0.0, 1.0)

        # Apply Upscaler_2 for shadow mask, light mask -> (B, 2, 10, H, W)
        if self.model_config.upsampler_sl == "unet":
            light_and_shadow_raw = self.up_scale_sl(encoder_features[1:], x)
            light_mask = torch.clip(
                F.relu(light_and_shadow_raw[:, 0, :, :, :]), 0.0, 1.0
            )
            shadow_mask = F.sigmoid(light_and_shadow_raw[:, 1, :, :, :])

        # Apply Upscaler_3 for occlusion mask, occlusion rgb -> (B, 4, 10, H, W)
        if self.model_config.upsampler_ob == "unet":
            occlusion_raw = self.up_scale_ob(encoder_features[1:], x)
            if not self.train_config.lambda_binary_occ:
                occlusion_mask = F.sigmoid(occlusion_raw[:, 0, :, :, :])
            else:
                occlusion_mask = occlusion_raw[:, 0, :, :, :]

            occlusion_rgb = F.sigmoid(occlusion_raw[:, 1:, :, :, :])

        return (
            gt_reconstruction,
            light_mask,
            shadow_mask,
            occlusion_mask,
            occlusion_rgb,
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
        shadow_light_mask_gt,
        occlusion_mask_gt,
    ):
        return self.loss(
            gt_reconstruction=gt_reconstruction,
            light_mask=light_mask,
            shadow_mask=shadow_mask,
            occlusion_mask=occlusion_mask,
            occlusion_rgb=occlusion_rgb,
            target=target,
            input=input,
            shadow_light_mask=shadow_light_mask_gt,
            occlusion_mask_gt=occlusion_mask_gt,
        )

    def training_step(self, batch, batch_idx):
        (
            x,
            y,
            sl,
            ob,
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
            light_mask = None
            shadow_mask = None
            occlusion_mask = None
            occlusion_rgb = None

        loss = self.loss_func(
            gt_reconstruction,
            light_mask,
            shadow_mask,
            occlusion_mask,
            occlusion_rgb,
            y,
            x,
            sl,
            ob,
        )

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        (
            x,
            y,
            sl,
            ob,
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
            light_mask = None
            shadow_mask = None
            occlusion_mask = None
            occlusion_rgb = None

        # loss, ob_recon = self.loss_func(
        loss = self.loss_func(
            gt_reconstruction,
            light_mask,
            shadow_mask,
            occlusion_mask,
            occlusion_rgb,
            y,
            x,
            sl,
            ob,
        )

        self.log("val_loss", loss, prog_bar=True, sync_dist=True)

        # Log images on the first validation step
        if (
            batch_idx == 0
            and not self.train_config.debug
            and self.current_epoch % self.train_config.log_img_every_n_epochs == 0
        ):
            pre_train_log_images(
                self.logger, gt_reconstruction, x
            ) if self.train_config.pre_train else log_images(
                self.logger,
                y,
                x,
                gt_reconstruction,
                light_mask,
                shadow_mask,
                occlusion_mask,
                occlusion_rgb,
                sl,
                ob,
            )
        self.validation_step_outputs.append(loss)
        return loss

    def test_step(self, batch, batch_idx):
        if len(batch) > 4:
            (
                x,
                y,
                sl,
                ob,
                dir,
            ) = batch  # --- x: (B, N, C, H, W), y: (B, C, H, W) | N: number of images in sequence
        else:
            (x, y, sl, ob) = batch
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
            sl,
            ob,
        )

        self.log("test_loss", loss, prog_bar=True)

        # Log evaluation images
        target = y.unsqueeze(2).repeat(1, 1, 10, 1, 1)  # tmp
        shadow_mask = shadow_mask.unsqueeze(1).repeat(1, 3, 1, 1, 1)
        light_mask = light_mask.unsqueeze(1).repeat(1, 3, 1, 1, 1)
        occlusion_mask = occlusion_mask.unsqueeze(1).repeat(1, 3, 1, 1, 1)

        ob_reconstruction = torch.where(
            occlusion_mask < 0.5,
            (target * 0),  # tmp
            occlusion_rgb,
        )

        if len(batch) > 4:
            evaluation_log_images(
                gt_reconstruction,
                ob_reconstruction,
                light_mask,
                shadow_mask,
                x,
                dir,
                self.eval_output,
            )

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

        if loss < self.best_val_loss:
            self.best_val_loss = loss
            if self.train_config.stage == "train_gt":
                torch.save(
                    self.up_scale_gt.state_dict(),
                    f"{self.log_dir}/up_scale_gt_model.pt",
                )
            if self.train_config.stage == "train_sl":
                torch.save(
                    self.up_scale_sl.state_dict(),
                    f"{self.log_dir}/up_scale_sl_model.pt",
                )
            if self.train_config.stage == "train_ob":
                torch.save(
                    self.up_scale_ob.state_dict(),
                    f"{self.log_dir}/up_scale_ob_model.pt",
                )

        self.validation_step_outputs.clear()
