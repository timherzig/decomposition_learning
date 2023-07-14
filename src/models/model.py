import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import lightning.pytorch as pl
import torch.nn.functional as F

from src.models.transformer.swin_transformer import SwinTransformer3D
from src.models.up_scaling.swin.upsampling import SwinTransformer3D_up
from src.models.up_scaling.unet.up_scale import UpSampler
from src.models.utils.utils import get_class

from utils.wandb_logging import (
    log_images,
    pre_train_log_images,
)
from einops import rearrange


class Decomposer(pl.LightningModule):
    def __init__(self, config, log_dir: str = None):
        super().__init__()

        self.data_config = config.data
        self.model_config = config.model
        self.train_config = config.train
        self.log_dir = log_dir

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
                pretrained=config.model.swin.checkpoint,
                patch_size=self.model_config.swin.patch_size,
                frozen_stages=1,
            )
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
                self.up_scale_gt.load_state_dict(
                    torch.load(self.model_config.unet_gt.checkpoint),
                )
                if self.model_config.unet_gt.freeze:
                    for param in self.up_scale_gt.parameters():
                        param.requires_grad = False

        elif self.model_config.upsampler_gt == "swin":
            self.decoder_gt_config = self.model_config.swin_gt.decoder
            arguments = dict(self.decoder_gt_config)
            self.up_scale_gt = SwinTransformer3D_up(**arguments)
            # self.up_scale_gt = SwinTransformer3D_up(out_chans=3, patch_size=self.model_config.swin.patch_size)

        # Shadow and light upsampling
        if self.model_config.upsampler_sl == "unet":
            self.decoder_sl_config = self.model_config.unet_sl.decoder
            arguments = dict(self.decoder_sl_config)
            self.up_scale_sl = UpSampler(**arguments)
            if self.model_config.unet_sl.checkpoint is not False:
                self.up_scale_sl.load_state_dict(
                    torch.load(self.model_config.unet_sl.checkpoint),
                )
                if self.model_config.unet_sl.freeze:
                    for param in self.up_scale_sl.parameters():
                        param.requires_grad = False

        elif self.model_config.upsampler_sl == "swin":
            self.decoder_sl_config = self.model_config.swin_sl.decoder
            arguments = dict(self.decoder_sl_config)
            self.up_scale_sl = SwinTransformer3D_up(**arguments)
            # self.up_scale_sl = SwinTransformer3D_up(out_chans=2, patch_size=self.model_config.swin.patch_size)

        # Object upsampling
        if self.model_config.upsampler_ob == "unet":
            self.decoder_ob_config = self.model_config.unet_ob.decoder
            arguments = dict(self.decoder_ob_config)
            self.up_scale_ob = UpSampler(**arguments)
            if self.model_config.unet_ob.checkpoint is not False:
                self.up_scale_ob.load_state_dict(
                    torch.load(self.model_config.unet_ob.checkpoint)
                )
                if self.model_config.unet_ob.freeze:
                    for param in self.up_scale_ob.parameters():
                        param.requires_grad = False

        elif self.model_config.upsampler_sl == "swin":
            self.decoder_ob_config = self.model_config.swin_ob.decoder
            arguments = dict(self.decoder_ob_config)
            self.up_scale_ob = SwinTransformer3D_up(**arguments)
            # self.up_scale_ob = SwinTransformer3D_up(out_chans=4, patch_size=self.model_config.swin.patch_size)

    def forward(self, x):
        if self.model_config.upsampler_gt == "swin":
            return self.forward_swin(x)
        else:
            return self.forward_unet(x)

    def forward_swin(self, x):
        x = self.swin.patch_embed(x)
        x = self.swin.pos_drop(x)

        for idx, layer in enumerate(self.swin.layers):
            x, _ = layer(x.contiguous())
        x = rearrange(x, "n c d h w -> n d h w c")
        x = self.swin.norm(x)
        x = rearrange(x, "n d h w c -> n c d h w")
        # print(x.shape)

        # Perform upsampling
        gt_reconstruction = self.up_scale_gt.forward(x)
        gt_reconstruction = torch.squeeze(gt_reconstruction)
        masks = self.up_scale_sl.forward(x)
        light_mask = masks[:, 0, :, :, :]
        shadow_mask = masks[:, 1, :, :, :]
        occlusion = self.up_scale_ob.forward(x)
        occlusion_mask = occlusion[:, 0, :, :, :]
        occlusion_rgb = occlusion[:, 1:, :, :, :]

        return (
            torch.clip(gt_reconstruction, -1.0, 1.0),
            torch.clip(light_mask, -1.0, 1.0),
            torch.clip(shadow_mask, -1.0, 1.0),
            torch.clip(occlusion_mask, -1.0, 1.0),
            torch.clip(occlusion_rgb, -1.0, 1.0),
        )

    def forward_unet(self, x):
        x, encoder_features = self.swin(x)
        if self.train_config.debug:
            print(f"swin x shape: {x.shape}")
            for idx, ef in enumerate(encoder_features):
                print(f"swin encoder_feature {idx} shape: {ef.shape}")

        # Apply Upscaler_1 for reconstruction -> (B, 3, H, W)
        if self.model_config.upsampler_gt == "unet":
            # Apply sigmoid activation layer
            gt_reconstruction = F.sigmoid(
                torch.squeeze(self.up_scale_gt(encoder_features[1:], x))
            )
            if self.train_config.pre_train:
                return torch.clip(gt_reconstruction, 0.0, 1.0)

        # Apply Upscaler_2 for shadow mask, light mask -> (B, 2, 10, H, W)
        if self.model_config.upsampler_sl == "unet":
            light_and_shadow_raw = self.up_scale_sl(encoder_features[1:], x)
            light_mask = F.relu(
                # self.up_scale_sl(encoder_features[1:], x)[:, 0, :, :, :]
                light_and_shadow_raw[:, 0, :, :, :]
            )  # ReLU activation
            shadow_mask = F.sigmoid(
                # self.up_scale_sl(encoder_features[1:], x)[:, 0, :, :, :]
                light_and_shadow_raw[:, 1, :, :, :]
            )  # Sigmoid activation

        # Apply Upscaler_3 for occlusion mask, occlusion rgb -> (B, 4, 10, H, W)
        if self.model_config.upsampler_ob == "unet":
            occlusion_raw = self.up_scale_ob(encoder_features[1:], x)
            occlusion_mask = F.relu(
                # self.up_scale_ob(encoder_features[1:], x)[:, 0, :, :, :]
                occlusion_raw[:, 0, :, :, :]
            )  # ReLU
            occlusion_rgb = F.relu(
                # self.up_scale_ob(encoder_features[1:], x)[:, 1:, :, :, :]
                occlusion_raw[:, 1:, :, :, :]
            )  # ReLU

        return (
            torch.clip(gt_reconstruction, 0.0, 1.0),
            torch.clip(light_mask, 0.0, 1.0),
            torch.clip(shadow_mask, 0.0, 1.0),
            torch.relu(occlusion_mask),
            torch.clip(occlusion_rgb, 0.0, 1.0),
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
            and (
                self.data_config.sanity_check
                and self.current_epoch % 100 == 0
                or not self.data_config.sanity_check
                and self.current_epoch % 10 == 0
            )
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

        # Log images on the first test step
        if batch_idx == 0 and not self.train_config.debug:
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
