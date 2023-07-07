import torch
import numpy as np

from PIL import Image, ImageChops, ImageFilter
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt
import cv2 
import os
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

    return torch.div(images.float(), 255.0)



def get_occlusion_gt(gt, occluded_gt, sl_masks):
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
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    fig.subplots_adjust(left=0, right=1, hspace=0, wspace=0.01)
    sl_masks = torch.moveaxis(sl_masks, 0, 3)

    oc_masks = []
    for idx, sl_mask in enumerate(sl_masks):
        occluded = np.array(occluded_gti[idx])
        occluded = occluded.astype(np.uint8)


        sl_mask = sl_mask.numpy()
        sl_mask = cv2.normalize(sl_mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        diff = np.array(ImageChops.subtract(to_pil(sl_mask), to_pil(occluded)))

        oc_masks.append(diff)

    oc_masks = torch.squeeze(torch.tensor(np.array(oc_masks)))
    oc_masks = torch.permute(oc_masks, (3, 0, 1, 2))
    print("OC MAKS: ", oc_masks.shape)

    return torch.div(oc_masks.float(), 255.0)