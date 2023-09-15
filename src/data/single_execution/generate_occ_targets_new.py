import os
import sys

from omegaconf import OmegaConf
import PIL
import cv2
from argparse import ArgumentParser
from tqdm import tqdm
from torchvision.transforms import ToPILImage, Grayscale
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
import torch

to_pil = ToPILImage()
to_gray = Grayscale()
ssim_metric = (
    SSIM(return_full_image=True).to("cuda")
    if torch.cuda.is_available()
    else SSIM(return_full_image=True)
)

sys.path.append(os.getcwd())
# print(sys.path)

from src.models.model import Decomposer
from src.models.utils.utils import images_to_tensor


def get_X_SL(input_sequence, model):
    (oi_reconstruction, light_mask, shadow_mask, _, _) = model(
        input_sequence.unsqueeze(0)
    )
    oi_reconstruction_repeated = oi_reconstruction.unsqueeze(1).repeat(1, 10, 1, 1)
    X_SL = (oi_reconstruction_repeated * shadow_mask) + light_mask
    X_SL = X_SL.swapaxes(0, 1)
    del oi_reconstruction_repeated
    del oi_reconstruction
    del light_mask
    del shadow_mask
    return X_SL


def get_mask_ssim(input_img, x_sl, threshold=0.3):
    ssim = ssim_metric(x_sl, input_img)
    ssim_img = ssim[1].squeeze(0)
    # invert ssim_img
    ssim_img = 1 - ssim_img
    # threshold ssim_img
    ssim_img[ssim_img < threshold] = 0
    ssim_img[ssim_img >= threshold] = 1
    return ssim_img


def get_mask_subtraction(
    input_img, x_sl, threshold=0.3, use_adaptive_thresholding=False
):
    extracted_occlusion = torch.abs(input_img - x_sl)
    # extracted_occlusion[extracted_occlusion < threshold] = 0
    # extracted_occlusion[extracted_occlusion >= threshold] = 1

    # adaptive thresholding
    if use_adaptive_thresholding:
        extracted_occlusion = extracted_occlusion.squeeze(0)
        extracted_occlusion = extracted_occlusion.detach().numpy() * 255
        extracted_occlusion = extracted_occlusion.astype("uint8")
        extracted_occlusion = extracted_occlusion.swapaxes(0, 1).swapaxes(1, 2)
        extracted_occlusion = cv2.cvtColor(extracted_occlusion, cv2.COLOR_RGB2GRAY)
        extracted_occlusion = cv2.adaptiveThreshold(
            extracted_occlusion,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            11,
            2,
        )
        extracted_occlusion = cv2.cvtColor(extracted_occlusion, cv2.COLOR_GRAY2RGB)
        extracted_occlusion = extracted_occlusion.swapaxes(1, 2).swapaxes(0, 1)
        extracted_occlusion = extracted_occlusion / 255.0
        extracted_occlusion = torch.from_numpy(extracted_occlusion).float()
    else:
        extracted_occlusion[extracted_occlusion < threshold] = 0
        extracted_occlusion[extracted_occlusion >= threshold] = 1

    return extracted_occlusion


def main(args):
    config_path = args.config

    config = OmegaConf.load(config_path)

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if config.train.device is None
        else config.train.device
    )

    with torch.no_grad():
        model = Decomposer.load_from_checkpoint(
            config.model.checkpoint, config=config, log_dir=None, map_location=device
        )

        model.eval()

        dataset = os.listdir(f"{config.data.path_to_data}/SIAR")
        # remove non-numeric elements
        dataset = [x for x in dataset if x.split(".")[0].isnumeric()]

        ssim_threshold = args.ssim_threshold
        subtraction_threshold = args.subtraction_threshold

        for sample_name in tqdm(dataset):
            if os.path.exists(
                f"{config.data.path_to_data}/SIAR_OCC_SSIM/{sample_name}"
            ) and os.path.exists(
                f"{config.data.path_to_data}/SIAR_OCC_SUB/{sample_name}"
            ):
                continue

            input_images = images_to_tensor(
                f"{config.data.path_to_data}/SIAR/{sample_name}"
            )
            input_images = input_images.to(device)
            # Get names for sequence elements
            sequence_elements = os.listdir(
                f"{config.data.path_to_data}/SIAR/{sample_name}"
            )
            sequence_elements.sort()
            # remove non-numeric elements
            sequence_elements = [
                x for x in sequence_elements if x.split(".")[0].isnumeric()
            ]

            # Get all X_SL for one sample sequence of dataset from model
            X_SL = get_X_SL(input_images, model)  # X_SL.shape = (10, 3, 256, 256)
            input_images = input_images.swapaxes(
                0, 1
            )  # input_images.shape = (10, 3, 256, 256)

            # Get masks
            ssim_img = get_mask_ssim(input_images, X_SL, ssim_threshold)
            subtraction_img = get_mask_subtraction(
                input_images, X_SL, subtraction_threshold
            )

            # save masks as png
            for elem_name, img_ssim, img_subtraction in zip(
                sequence_elements, ssim_img, subtraction_img
            ):
                # save ssim mask using PIL
                img_ssim = img_ssim.detach().cpu().numpy()
                img_ssim = img_ssim.swapaxes(0, 1).swapaxes(1, 2)
                img_ssim = img_ssim * 255
                img_ssim = img_ssim.astype("uint8")
                img_ssim = to_gray(PIL.Image.fromarray(img_ssim))
                # create folder if not exists
                os.makedirs(
                    f"{config.data.path_to_data}/SIAR_OCC_SSIM/{sample_name}",
                    exist_ok=True,
                )
                img_ssim.save(
                    f"{config.data.path_to_data}/SIAR_OCC_SSIM/{sample_name}/{elem_name}",
                    "PNG",
                )

                # save subtraction mask using PIL
                img_subtraction = img_subtraction.detach().cpu().numpy()
                img_subtraction = img_subtraction.swapaxes(0, 1).swapaxes(1, 2)
                img_subtraction = img_subtraction * 255
                img_subtraction = img_subtraction.astype("uint8")
                img_subtraction = to_gray(PIL.Image.fromarray(img_subtraction))
                # create folder if not exists
                os.makedirs(
                    f"{config.data.path_to_data}/SIAR_OCC_SUB/{sample_name}",
                    exist_ok=True,
                )
                img_subtraction.save(
                    f"{config.data.path_to_data}/SIAR_OCC_SUB/{sample_name}/{elem_name}",
                    "PNG",
                )


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        help="Path to config.yaml file.",
        default="config/occ_generation_w_model.yaml",
    )

    parser.add_argument(
        "--ssim_threshold",
        type=int,
        help="Threshold for setting SSIM mask to 0 or 1. SSIM above threshold is set to 1, SSIM below threshold is set to 0.",
        default=0.4,
    )

    parser.add_argument(
        "--subtraction_threshold",
        type=int,
        help="Threshold for setting subtraction mask to 0 or 1. AbsDifference above threshold is set to 1, AbsDifference below threshold is set to 0.",
        default=0.1,
    )

    args = parser.parse_args()

    main(args)
