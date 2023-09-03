import torch
import torch.nn as nn
from torch.nn import MSELoss

from src.models.utils.utils import get_class, get_pos_weight
from src.models.metrics import MSE, SSIM


def weight_decay(model, weight_decay):
    """Weight decay. Sum over all weights squared.

    Args:
        model (torch.nn.Module): Model with parameters to be summed over
        weight_decay (float): Weight decay parameter

    Returns:
        torch.Tensor: Sum of all weights squared * weight_decay
    """

    if weight_decay == 0.0 or weight_decay is None:
        return 0.0

    decay = 0.0
    for _, param in model.named_parameters():
        # if parameter ist trainable
        if param.requires_grad:
            decay += torch.sum(param**2)
    return decay * weight_decay


def mask_decay(mask, mask_decay):
    """Sum over all mask elements squared

    Args:
        mask (torch.Tensor): Mask to be summed over
        mask_decay (float): Mask decay parameter

    Returns:
        torch.Tensor: Sum of all mask elements squared * mask_decay
    """
    if mask_decay == 0.0 or mask_decay is None:
        return 0.0

    return mask_decay * torch.sum(mask**2)


def get_metric(metric):
    return get_class(metric, ["src.models.metrics"])


class base_loss:
    """Base loss. Sum of MSE between (i) and (ii) \\
        (i) reconstructed occluded image and input \\
        (ii) gt (unoccluded) image and the respective model gt prediction.
    """

    def __init__(self, model, config):
        super().__init__()

        self.model = model
        self.config = config
        metric_class = get_metric(self.config.metric)
        self.metric = metric_class()

    def __call__(
        self,
        gt_reconstruction,
        light_mask,
        shadow_mask,
        occlusion_mask,
        occlusion_rgb,
        target,
        input,
        shadow_light_mask,
        occlusion_mask_gt,
    ):
        gt_loss = self.metric(gt_reconstruction, target)

        gt_reconstruction = gt_reconstruction.unsqueeze(2).repeat(1, 1, 10, 1, 1)
        shadow_mask = shadow_mask.unsqueeze(1).repeat(1, 3, 1, 1, 1)
        light_mask = light_mask.unsqueeze(1).repeat(1, 3, 1, 1, 1)
        occlusion_mask = occlusion_mask.unsqueeze(1).repeat(1, 3, 1, 1, 1)

        reconstruction = torch.where(
            occlusion_mask < 0.5,
            (gt_reconstruction * shadow_mask + light_mask),
            occlusion_rgb,
        )

        reconstruction_loss = self.metric(reconstruction, input)

        return (
            gt_loss
            + reconstruction_loss
            + weight_decay(self.model, self.config.weight_decay)
        )


class reconstruction_loss:
    """Reconstruction loss. MSE between reconstructed occluded image and input."""

    def __init__(self, model, config):
        super().__init__()

        self.model = model
        self.config = config
        metric_class = get_metric(self.config.metric_all)
        self.metric_rec = metric_class()

    def __call__(
        self,
        gt_reconstruction,
        light_mask,
        shadow_mask,
        occlusion_mask,
        occlusion_rgb,
        target,
        input,
        shadow_light_mask,
        occlusion_mask_gt,
    ):
        gt_reconstruction = gt_reconstruction.unsqueeze(2).repeat(1, 1, 10, 1, 1)
        shadow_mask = shadow_mask.unsqueeze(1).repeat(1, 3, 1, 1, 1)
        light_mask = light_mask.unsqueeze(1).repeat(1, 3, 1, 1, 1)
        occlusion_mask = occlusion_mask.unsqueeze(1).repeat(1, 3, 1, 1, 1)

        reconstruction = torch.where(
            occlusion_mask < 0.5,
            (gt_reconstruction * shadow_mask + light_mask),
            occlusion_rgb,
        )

        reconstruction_loss = self.metric_rec(reconstruction, input)

        return reconstruction_loss + weight_decay(self.model, self.config.weight_decay)


class occlusion_reconstruction_loss:
    """Reconstruction loss. MSE between reconstructed occluded image and input."""

    def __init__(self, model, config):
        super().__init__()

        self.model = model
        self.config = config
        metric_class = get_metric(self.config.metric_all)
        self.metric_rec = metric_class()
        self.metric_class_occ = get_metric(self.config.metric_occ_mask)

    def __call__(
        self,
        gt_reconstruction,
        light_mask,
        shadow_mask,
        occlusion_mask,
        occlusion_rgb,
        target,
        input,
        shadow_light_mask,
        occlusion_mask_gt,
    ):
        # gt_reconstruction = gt_reconstruction.unsqueeze(2).repeat(1, 1, 10, 1, 1)
        # shadow_mask = shadow_mask.unsqueeze(1).repeat(1, 3, 1, 1, 1)
        # light_mask = light_mask.unsqueeze(1).repeat(1, 3, 1, 1, 1)

        # Occlusion mask loss
        if self.config.lambda_binary_occ:
            pos_weight = get_pos_weight(occlusion_mask_gt)
            metric_occ = self.metric_class_occ(pos_weight=pos_weight)
            binary_occ_mask_loss = self.config.lambda_binary_occ * metric_occ(
                occlusion_mask, occlusion_mask_gt
            )
        else:
            binary_occ_mask_loss = 0

        # mask decay
        md = mask_decay(occlusion_mask, self.config.mask_decay)

        wd = weight_decay(self.model, self.config.weight_decay)

        # Reconstruction loss
        occlusion_mask = occlusion_mask.unsqueeze(1).repeat(1, 3, 1, 1, 1)

        reconstruction = torch.where(
            occlusion_mask < 0.9,
            shadow_light_mask,
            occlusion_rgb,
        )
        reconstruction_loss = self.metric_rec(reconstruction, input)

        # reconstruction = torch.where(
        #     occlusion_mask < 0.5,
        #     (gt_reconstruction * shadow_mask + light_mask),
        #     occlusion_rgb,
        # )
        return reconstruction_loss + binary_occ_mask_loss + md + wd


class pre_train_loss:
    """Pre-train loss. MSE between gt (unoccluded) image and the respective model gt prediction."""

    def __init__(self, model, config):
        super().__init__()

        self.model = model
        self.config = config
        metric_class = get_metric(self.config.metric_all)
        self.metric = metric_class()

    def __call__(
        self,
        gt_reconstruction,
        light_mask,
        shadow_mask,
        occlusion_mask,
        occlusion_rgb,
        target,
        input,
        shadow_light_mask,
        occlusion_mask_gt,
    ):
        gt_loss = self.metric(gt_reconstruction, input)

        return gt_loss + weight_decay(self.model, self.config.weight_decay)


class separate_head_loss:
    def __init__(self, model, config):
        super().__init__()

        self.model = model
        self.config = config
        metric_class_gt = get_metric(self.config.metric_gt)
        metric_class_sl = get_metric(self.config.metric_sl)
        metric_class_ob = get_metric(self.config.metric_ob)
        metric_class_all = get_metric(self.config.metric_all)

        self.metric_gt = metric_class_gt()
        self.metric_sl = metric_class_sl()
        self.metric_ob = metric_class_ob()
        self.metric_all = metric_class_all()

    def __call__(
        self,
        gt_reconstruction,
        light_mask,
        shadow_mask,
        occlusion_mask,
        occlusion_rgb,
        target,
        input,
        shadow_light_mask,
        occlusion_mask_gt,
    ):
        if self.config.stage == "train_gt":
            return self.metric_gt(gt_reconstruction, target) + weight_decay(
                self.model, self.config.weight_decay
            )

        if self.config.stage == "train_sl":
            target = target.unsqueeze(2).repeat(1, 1, 10, 1, 1)
            shadow_mask = shadow_mask.unsqueeze(1).repeat(1, 3, 1, 1, 1)
            light_mask = light_mask.unsqueeze(1).repeat(1, 3, 1, 1, 1)
            sl_reconstruction = target * shadow_mask + light_mask
            return self.metric_sl(sl_reconstruction, shadow_light_mask) + weight_decay(
                self.model, self.config.weight_decay
            )

        if self.config.stage == "train_ob":
            target = target.unsqueeze(2).repeat(1, 1, 10, 1, 1)
            shadow_mask = shadow_mask.unsqueeze(1).repeat(1, 3, 1, 1, 1)
            light_mask = light_mask.unsqueeze(1).repeat(1, 3, 1, 1, 1)
            occlusion_mask = occlusion_mask.unsqueeze(1).repeat(1, 3, 1, 1, 1)
            ob_reconstruction = torch.where(
                occlusion_mask < 0.5,
                target * shadow_mask + light_mask,
                occlusion_rgb,
            )
            return self.metric_ob(ob_reconstruction, occlusion_mask_gt) + weight_decay(
                self.model, self.config.weight_decay
            )

        if self.config.stage == "train_all_heads":
            gt_loss = self.metric_gt(gt_reconstruction, target)

            shadow_mask = shadow_mask.unsqueeze(1).repeat(1, 3, 1, 1, 1)
            light_mask = light_mask.unsqueeze(1).repeat(1, 3, 1, 1, 1)
            target = target.unsqueeze(2).repeat(1, 1, 10, 1, 1)
            sl_reconstruction = target * shadow_mask + light_mask
            sl_loss = self.metric_sl(sl_reconstruction, shadow_light_mask)

            occlusion_mask = occlusion_mask.unsqueeze(1).repeat(1, 3, 1, 1, 1)
            ob_reconstruction = torch.where(
                occlusion_mask < 0.5,
                (target * shadow_mask + light_mask),
                occlusion_rgb,
            )
            # ob_reconstruction = torch.clip(ob_reconstruction, 0.0, 1.0)
            ob_loss = self.metric_ob(ob_reconstruction, input)

            return (
                gt_loss
                + sl_loss
                + ob_loss
                + weight_decay(self.model, self.config.weight_decay)
            )

        if self.config.stage == "train_all":
            gt_loss = self.metric_gt(gt_reconstruction, target)  # tmp
            target = target.unsqueeze(2).repeat(1, 1, 10, 1, 1)  # tmp
            shadow_mask = shadow_mask.unsqueeze(1).repeat(1, 3, 1, 1, 1)
            light_mask = light_mask.unsqueeze(1).repeat(1, 3, 1, 1, 1)
            occlusion_mask = occlusion_mask.unsqueeze(1).repeat(1, 3, 1, 1, 1)

            ob_reconstruction = torch.where(
                occlusion_mask < 0.5,
                (target * shadow_mask + light_mask),  # tmp
                occlusion_rgb,
            )

            return (
                self.metric_all(ob_reconstruction, input)
                + gt_loss * self.config.lambda_gt_loss  # tmp
                + weight_decay(self.model, self.config.weight_decay)
            )


class occ_binary_pretraining_loss:
    def __init__(self, model, config):
        super().__init__()

        self.model = model
        self.config = config
        self.metric_occ_mask = get_metric(self.config.metric_occ_mask)
        self.metric_occ_rgb = get_metric(self.config.metric_occ_rgb)
        self.metric_mask = self.metric_occ_mask()
        self.metric_rgb = self.metric_occ_rgb()

    def __call__(
        self,
        gt_reconstruction,
        light_mask,
        shadow_mask,
        occlusion_mask,
        occlusion_rgb,
        target,
        input,
        shadow_light_mask,
        occlusion_mask_gt,
    ):
        pos_weight = get_pos_weight(occlusion_mask_gt)

        # always reinitialize the BCEWithLogitsLoss with new positive weights
        metric_mask = self.metric_occ_mask(pos_weight=pos_weight)

        loss_mask = metric_mask(occlusion_mask, occlusion_mask_gt)

        return loss_mask  # + weight_decay(self.model, self.config.weight_decay)
