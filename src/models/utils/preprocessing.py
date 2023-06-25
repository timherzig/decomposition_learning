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

    return np.array(sl_image)


def get_shadow_light_gt(gt, occluded_gt):
    """
    Returns a prediction of the occluded images without the occlusions.
    So they only contain shadow and light.

    Parameters:
        gt (torch.Tensor): Ground truth image. Shape: (3, H, W)
        occluded_gt (torch.Tensor): Occluded ground truth images. Shape: (3, 10, H, W)
    """

    to_pil = ToPILImage()

    images = []

    gti = to_pil(gt)
    occluded_gti = [
        to_pil(occluded_gt[:, j, :, :]) for j in range(occluded_gt.shape[1])
    ]

    sl_mask = [shadow_light_mask(gti, x) for x in occluded_gti]
    images.append(sl_mask)

    images = torch.squeeze(torch.tensor(np.array(images)))
    images = torch.permute(images, (3, 0, 1, 2))

    return images.float()
