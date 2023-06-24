import torch
import numpy as np

from PIL import Image, ImageChops, ImageFilter
from torchvision.transforms import ToPILImage


def shadow_light_mask(gt, occluded):
    """
    Returns a prediction of the occluded images without the occlusions.
    """

    light = ImageChops.subtract(occluded, gt)
    light = light.filter(ImageFilter.GaussianBlur(radius=20))

    shadow = ImageChops.invert(ImageChops.subtract(gt, occluded))
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=20))

    sl_image = ImageChops.multiply(
        ImageChops.add(gt, light.convert("RGB")), shadow.convert("RGB")
    )
    return sl_image


def get_shadow_light_gt(gt, occluded_gt):
    """
    Returns a prediction of the occluded images without the occlusions.
    So they only contain shadow and light.

    Parameters:
        gt (torch.Tensor): Ground truth image. Shape: (B, 3, H, W)
        occluded_gt (torch.Tensor): Occluded ground truth images. Shape: (B, 10, 3, H, W)
    """

    to_pil = ToPILImage()

    batch_images = np.array([])

    for i in range(gt.shape[0]):
        gti = to_pil(gt[i, :, :, :])
        occluded_gti = [
            to_pil(occluded_gt[i, j, :, :, :]) for j in range(occluded_gt.shape[1])
        ]

        batch_images.append(
            np.array([np.array(shadow_light_mask(gti, x) for x in occluded_gti)])
        )

    return torch.tensor(batch_images)