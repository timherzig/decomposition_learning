import torch
from torch.nn import MSELoss
import torch.nn as nn


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


class base_loss(nn.Module):
    """Base loss. Sum of MSE between (i) and (ii) \\
        (i) reconstructed occluded image and input \\
        (ii) gt (unoccluded) image and the respective model gt prediction.
    """

    def __init__(self, model, config):
        super(base_loss, self).__init__()

        self.model = model
        self.config = config

    def forward(
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

        return (
            gt_loss
            + reconstruction_loss
            + weight_decay(self.model, self.config.weight_decay_param)
        )


class reconstruction_loss(nn.Module):
    """Reconstruction loss. MSE between reconstructed occluded image and input."""

    def __init__(self, model, config):
        super(reconstruction_loss, self).__init__()

        self.model = model
        self.config = config

    def forward(
        self,
        gt_reconstruction,
        light_mask,
        shadow_mask,
        occlusion_mask,
        occlusion_rgb,
        input,
    ):
        loss = MSELoss()
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

        return reconstruction_loss + weight_decay(
            self.model, self.config.weight_decay_param
        )


class pre_train_loss(nn.Module):
    """Pre-train loss. MSE between gt (unoccluded) image and the respective model gt prediction."""

    def __init__(self, model, config):
        super(pre_train_loss, self).__init__()

        self.model = model
        self.config = config

    def forward(
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
        gt_loss = loss(gt_reconstruction, input)

        return gt_loss + weight_decay(self.model, self.config.weight_decay_param)


class regularized_loss(nn.Module):
    """Regularized loss. \\
    > Loss =  l_1 * gt_loss + l_2 * decomp_loss + weight_decay + mask_decay - l_3 * Occ_diff
    > where occlusion_mas = 1: Occ_diff = || Occ_RGB - gt_RGB ||_2 (*l_3 or clip range)
    """

    def __init__(self, model, config):
        super(regularized_loss, self).__init__()

        self.model = model
        self.config = config

    def forward(
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
        og_occ_mask = occlusion_mask
        occlusion_mask = occlusion_mask.unsqueeze(1).repeat(1, 3, 1, 1, 1)

        decomposition_reconstruction = torch.where(
            occlusion_mask < 0.5,
            (gt_reconstruction * shadow_mask + light_mask),
            occlusion_rgb,
        )

        decomp_loss = loss(decomposition_reconstruction, input)

        occ_diff = torch.norm(
            occlusion_mask * gt_reconstruction - occlusion_mask * occlusion_rgb, 2
        )

        final_loss = (
            self.config.lambda_gt_loss * gt_loss
            + self.config.lambda_decomp_loss * decomp_loss
            + weight_decay(self.model, self.config.weight_decay_param)
            + mask_decay(og_occ_mask, self.config.mask_decay_param)
            - self.config.lambda_occlusion_difference * occ_diff
        )

        return final_loss
