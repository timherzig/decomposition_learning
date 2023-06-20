import torch
from torch.nn import MSELoss


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


def base_loss(
    gt_reconstruction,
    light_mask,
    shadow_mask,
    occlusion_mask,
    occlusion_rgb,
    target,
    input,
    model,
    weight_decay_param,
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

    return gt_loss + reconstruction_loss + weight_decay(model, weight_decay_param)


def reconstruction_loss(
    gt_reconstruction,
    light_mask,
    shadow_mask,
    occlusion_mask,
    occlusion_rgb,
    input,
    model,
    weight_decay_param,
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

    return reconstruction_loss + weight_decay(model, weight_decay_param)


def pre_train_loss(gt_reconstruction, input, model, weight_decay_param):
    loss = MSELoss()
    gt_loss = loss(gt_reconstruction, input)
    return gt_loss + weight_decay(model, weight_decay_param)


def regularized_loss(
    gt_reconstruction,
    light_mask,
    shadow_mask,
    occlusion_mask,
    occlusion_rgb,
    target,
    input,
    model,
    weight_decay_param,
    mask_decay_param,
    lambda_gt_loss,
    lambda_decomp_loss,
    lambda_occlusion_difference,
):
    """Compute the regularized loss
        > Loss =  l_1 * gt_loss + l_2 * decomp_loss + weight_decay + mask_decay - l_3 * Occ_diff
        > where occlusion_mas = 1: Occ_diff = || Occ_RGB - gt_RGB ||_2 (*l_3 or clip range)

    Args:
        gt_reconstruction (torch.Tensor): Ground truth reconstruction. Shape (B, 3, H, W)
        light_mask (torch.Tensor): Light mask. Shape (B, 10, 1, H, W)
        shadow_mask (torch.Tensor): Shadow mask. Shape (B, 10, 1, H, W)
        occlusion_mask (torch.Tensor): Occlusion mask. Shape (B, 10, 1, H, W)
        occlusion_rgb (torch.Tensor): Occlusion rgb. Shape (B, 10, 3, H, W)
        target (torch.Tensor): Target image. Shape (B, 3, H, W)
        input (torch.Tensor): Input image sequence. Shape (B, 10, 3, H, W)
        model (torch.nn.Module): Model with parameters for weight decay
        weight_decay_param (float): Weight decay parameter
        mask_decay_param (float): Mask decay parameter
        lambda_gt_loss (float): Weight for gt_loss
        lambda_decomp_loss (float): Weight for decomp_loss
        lambda_occlusion_difference (float): Weight for occ_diff

    Returns:
        torch.Tensor: Regularized loss
    """
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
        lambda_gt_loss * gt_loss
        + lambda_decomp_loss * decomp_loss
        + weight_decay(model, weight_decay_param)
        + mask_decay(og_occ_mask, mask_decay_param)
        - lambda_occlusion_difference * occ_diff
    )

    return final_loss