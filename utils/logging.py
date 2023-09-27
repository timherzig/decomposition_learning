import cv2
import os
import numpy as np
from torchvision.transforms import ToPILImage


def evaluation_log_images(
    occ_reconstruction,
    gt_reconstruction,
    decomp_reconstruction,
    light_mask,
    shadow_mask,
    x,
    sequence_dirs,
    output_folder,
):
    to_pil = ToPILImage()

    os.makedirs(output_folder, exist_ok=True)
    for idx, dir_name in enumerate(sequence_dirs):
        sequence_path = os.path.join(output_folder, dir_name)
        gt_recon = gt_reconstruction[idx, :, :, :].to("cpu")
        occ_bin_masks = occ_reconstruction[idx, :, :, :, :].to("cpu")
        light_masks = light_mask[idx, :, :, :, :].to("cpu")
        shadow_masks = shadow_mask[idx, :, :, :, :].to("cpu")
        occ_gt = x[idx, :, :, :, :].to("cpu")
        decomp = decomp_reconstruction[idx, :, :, :, :].to("cpu")

        gt_recon_path = os.path.join(sequence_path, "gt_reconstruction.png")
        os.makedirs(sequence_path, exist_ok=True)
        to_pil(gt_recon).save(gt_recon_path)
        for i in range(10):
            gt_occluded = occ_gt[:, i, :, :].squeeze()

            occ_bin = occ_bin_masks[:, i, :, :].squeeze()
            occ_path = os.path.join(sequence_path, "occ_" + str(i + 1) + ".png")
            to_pil(occ_bin).save(occ_path)

            light = light_masks[:, i, :, :].squeeze()
            light_path = os.path.join(sequence_path, "light_" + str(i + 1) + ".png")
            to_pil(light).save(light_path)

            shadow = shadow_masks[:, i, :, :].squeeze()
            shadow_path = os.path.join(sequence_path, "shadow_" + str(i + 1) + ".png")
            to_pil(shadow).save(shadow_path)

            mult = gt_occluded * shadow
            mult_path = os.path.join(sequence_path, "mult_" + str(i + 1) + ".png")
            to_pil(mult).save(mult_path)

            decomp_path = os.path.join(sequence_path, "decomp_" + str(i + 1) + ".png")
            decomp_i = decomp[:, i, :, :].squeeze().clip(max=1)

            to_pil(decomp_i).save(decomp_path)


