import torch
import numpy as np

from PIL import Image, ImageChops, ImageFilter
from torchvision.transforms import ToPILImage
import cv2
import os
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
import torchvision.transforms as transforms

ssim_metric = SSIM(return_full_image=True)


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


def adaptiveThresholding(torch_img, upscale):
    # Upscale to range 0 - 255
    # First normalize to 0 - 1
    if upscale:
        torch_img = (
            (torch_img - torch_img.min()) / (torch_img.max() - torch_img.min()) * 255
        )

    for_blur = torch_img.numpy().astype(np.uint8)

    blur = cv2.GaussianBlur(for_blur, (5, 5), 0)
    _, img_thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img_thresh = img_thresh / 255
    return torch.from_numpy(img_thresh)


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


def get_occlusion_gt(gt, images):
    """
    Returns a prediction of the occluded images without the occlusions.
    So they only contain shadow and light.

    Parameters:
        gt (torch.Tensor): Ground truth image. Shape: (3, H, W)
        occluded_gt (torch.Tensor): Occluded ground truth images. Shape: (3, 10, H, W)
    """
    transform = transforms.Compose([transforms.PILToTensor()])
    toGrayscale = transforms.Grayscale()
    toPIL = transforms.ToPILImage()

    # read the images and gt and convert to numpy arrays
    # images_PIL = [Image.open(x) for x in images]
    # images = [transform(x) for x in images_PIL]
    # images = torch.stack(images).swapaxes(0, 1)

    # gt = transform(Image.open(gt))

    # get images without occlusions (gt with shadow and light only)
    sl_images = get_shadow_light_gt(gt, images)
    # sl_images_PIL = [transforms.ToPILImage()(x) for x in sl_images.swapaxes(0, 1)]

    # show all images in a grid
    # ger random images
    # idx = torch.randint(0, images.shape[1], (1,))

    # --- Experiment settings ---
    normalize = False
    divide_by_255 = True
    grayscale = True
    segmentation_threshold = 0.9
    # ---------------------------

    oc_masks = torch.zeros(sl_images.shape)
    for i in range(images.shape[1]):
        img = (
            images[:, i, :, :].unsqueeze(0).type(torch.FloatTensor)
        )  # img.mean() ~ 140 (or in that order of magnitude)!
        img_rgb = img.clone()
        # sl = sl_images[:, i, :, :].unsqueeze(0).type(torch.FloatTensor) # sl.mean() ~ 0.5 (or in that order of magnitude)!!

        # if divide_by_255:
        #     # --- Experiment 1 ---
        #     # devide by 255
        #     img = img / 255.0

        # if normalize:
        #     # --- Experiment 2 ---
        #     # normalize img and sl
        #     img = (img - img.min()) / (img.max() - img.min())
        if grayscale:
            # --- Experiment 3 ---
            # Directly improves SSIM but ***needs*** Experiment 1 or 2, both yield similar results
            # convert to dtype float
            img = toGrayscale(img)

        ssim_imgs = torch.zeros((10, 1, 256, 256))
        for j in range(sl_images.shape[1]):
            sl = (
                sl_images[:, j, :, :].unsqueeze(0).type(torch.FloatTensor)
            )  # sl.mean() ~ 0.5 (or in that order of magnitude)!!

            if normalize:
                # --- Experiment 2 ---
                # normalize img and sl
                a = 1

                sl = (sl - sl.min()) / (sl.max() - sl.min())

            if grayscale:
                # --- Experiment 3 ---
                # Directly improves SSIM but ***needs*** Experiment 1 or 2, both yield similar results
                # convert to dtype float
                sl = toGrayscale(sl)

            # compute patchwise SSIM
            ssim = ssim_metric(sl, img)
            ssim_img = ssim[1].squeeze(0)
            # invert ssim_img
            ssim_img = 1 - ssim_img
            # threshold ssim_img
            ssim_img_threshold = adaptiveThresholding(ssim_img[0], True).unsqueeze(0)
            ssim_imgs[j] = ssim_img_threshold

        ssim_img = torch.mean(ssim_imgs, axis=0)
        # ssim_img_threshold = adaptiveThresholding(ssim_img[0], True).unsqueeze(0)
        ssim_img[ssim_img < 1] = 0
        # --- Visualize ---
        # ssim tensor to PIL
        zeros = torch.zeros(img_rgb.shape)
        oc_mask = torch.where(ssim_img > 0.5, img_rgb, zeros)
        oc_masks[:, i, :, :] = oc_mask
    return oc_masks
