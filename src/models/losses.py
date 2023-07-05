import torch
import torch.nn as nn
from torch.nn import MSELoss

from src.models.utils.utils import get_class


def weight_decay(model, weight_decay):
    """Weight decay. Sum over all weights squared.

    Args:
        model (torch.nn.Module): Model with parameters to be summed over
        weight_decay (float): Weight decay parameter

    Returns:
        torch.Tensor: Sum of all weights squared * weight_decay
    """

    decay = 0.0
    for name, param in model.named_parameters():
        if name in ["weights"]:
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
        metric_class = get_metric(self.config.metric)
        self.metric = metric_class()

    def __call__(
        self,
        gt_reconstruction,
        light_mask,
        shadow_mask,
        occlusion_mask,
        occlusion_rgb,
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

        reconstruction_loss = self.metric(reconstruction, input)

        return reconstruction_loss + weight_decay(self.model, self.config.weight_decay)


class pre_train_loss:
    """Pre-train loss. MSE between gt (unoccluded) image and the respective model gt prediction."""

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
        gt_loss = self.metric(gt_reconstruction, input)

        return gt_loss + weight_decay(self.model, self.config.weight_decay)


class regularized_loss:
    """Regularized loss. \\
    > Loss =  l_1 * gt_loss + l_2 * decomp_loss + weight_decay + mask_decay - l_3 * Occ_diff
    > where occlusion_mas = 1: Occ_diff = || Occ_RGB - gt_RGB ||_2 (*l_3 or clip range)
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
        og_occ_mask = occlusion_mask
        occlusion_mask = occlusion_mask.unsqueeze(1).repeat(1, 3, 1, 1, 1)

        decomposition_reconstruction = torch.where(
            occlusion_mask < 0.5,
            (gt_reconstruction * shadow_mask + light_mask),
            occlusion_rgb,
        )

        decomp_loss = self.metric(decomposition_reconstruction, input)

        occ_diff = torch.norm(
            occlusion_mask * gt_reconstruction - occlusion_mask * occlusion_rgb, 2
        )

        final_loss = (
            self.config.lambda_gt_loss * gt_loss
            + self.config.lambda_decomp_loss * decomp_loss
            + weight_decay(self.model, self.config.weight_decay)
            + mask_decay(og_occ_mask, self.config.mask_decay)
            - self.config.lambda_occlusion_difference * occ_diff
        )

        return final_loss


class light_and_shadow_loss:
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

        imgs_no_occlusion_reconstruction = gt_reconstruction * shadow_mask + light_mask

        print(f"shadow_light_mask shape: {shadow_light_mask.shape}")
        print(f"imgs_no_occ shape: {imgs_no_occlusion_reconstruction.shape}")

        light_shadow_loss = self.metric(
            shadow_light_mask, imgs_no_occlusion_reconstruction
        )

        return (
            gt_loss
            + light_shadow_loss
            + weight_decay(self.model, self.config.weight_decay)
        )


class stage_loss:
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
        loss = 0

        if self.config.loss_stage == 1:
            return self.metric(gt_reconstruction, target) + weight_decay(
                self.model, self.config.weight_decay
            )

        target = target.unsqueeze(2).repeat(1, 1, 10, 1, 1)
        shadow_mask = shadow_mask.unsqueeze(1).repeat(1, 3, 1, 1, 1)
        light_mask = light_mask.unsqueeze(1).repeat(1, 3, 1, 1, 1)
        gt_reconstruction = gt_reconstruction.unsqueeze(2).repeat(1, 1, 10, 1, 1)

        if self.config.loss_stage == 2:
            sl_reconstruction = gt_reconstruction * shadow_mask + light_mask
            return self.metric(sl_reconstruction, shadow_light_mask) + weight_decay(
                self.model, self.config.weight_decay
            )

        occlusion_mask = occlusion_mask.unsqueeze(1).repeat(1, 3, 1, 1, 1)
        if self.config.loss_stage == 3:
            or_reconstruction = torch.where(
                occlusion_mask < 0.5,
                (gt_reconstruction * shadow_mask + light_mask),
                occlusion_rgb,
            )
            return (
                self.metric(or_reconstruction, input)
                + torch.mean(occlusion_mask)
                + weight_decay(self.model, self.config.weight_decay)
            )

        or_reconstruction = torch.where(
            occlusion_mask < 0.5,
            (gt_reconstruction * shadow_mask + light_mask),
            occlusion_rgb,
        )
        loss = self.metric(or_reconstruction, input) + torch.mean(occlusion_mask)

        return loss + weight_decay(self.model, self.config.weight_decay)


class separate_head_loss:
    def __init__(self, model, config):
        super().__init__()

        self.model = model
        self.config = config
        metric_class_gt = get_metric(self.config.metric_gt)
        metric_class_sl = get_metric(self.config.metric_sl)
        metric_class_ob = get_metric(self.config.metric_ob)

        self.metric_gt = metric_class_gt()
        self.metric_sl = metric_class_sl()
        self.metric_ob = metric_class_ob()

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
            sl_reconstruction = target * shadow_mask + light_mask
            return self.metric_sl(sl_reconstruction, shadow_light_mask) + weight_decay(
                self.model, self.config.weight_decay
            )

        if self.config.stage == "train_ob":
            target = target.unsqueeze(2).repeat(1, 1, 10, 1, 1)
            occlusion_mask = occlusion_mask.unsqueeze(1).repeat(1, 3, 1, 1, 1)
            ob_reconstruction = torch.where(
                occlusion_mask < 0.5,
                (target * shadow_mask + light_mask),
                occlusion_rgb,
            )
            return self.metric_ob(ob_reconstruction, occlusion_mask_gt) + weight_decay(
                self.model, self.config.weight_decay
            )

        if self.config.stage == "train_all":
            gt_loss = self.metric_gt(gt_reconstruction, target)

            target = target.unsqueeze(2).repeat(1, 1, 10, 1, 1)
            sl_reconstruction = target * shadow_mask + light_mask
            sl_loss = self.metric_sl(sl_reconstruction, shadow_light_mask)

            occlusion_mask = occlusion_mask.unsqueeze(1).repeat(1, 3, 1, 1, 1)
            ob_reconstruction = torch.where(
                occlusion_mask < 0.5,
                (target * shadow_mask + light_mask),
                occlusion_rgb,
            )
            ob_loss = self.metric_ob(ob_reconstruction, occlusion_mask_gt)

            return (
                gt_loss
                + sl_loss
                + ob_loss
                + weight_decay(self.model, self.config.weight_decay)
            )
