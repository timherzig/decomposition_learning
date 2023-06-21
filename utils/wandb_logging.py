import torch
from torchvision.transforms import ToPILImage
import wandb


def log_images(
    logger,
    y,
    x,
    gt_reconstruction,
    shadow_mask,
    light_mask,
    occlusion_mask,
    occlusion_rgb,
):
    """
    Logs one image sequence to the wandb logger

    params:
        y: unoccluded original input (1)
        x: occluded input (10)
        gt_reconstruction: ground truth reconstruction (1)
        shadow_mask: shadow mask (10)
        light_mask: light mask (10)
        occlusion_mask: occlusion mask (10)
        occlusion_rgb: occlusion rgb (10)
    """
    idx = torch.randint(0, y.shape[0], (1,)).item()

    y = y[idx, :, :, :]
    x = x[idx, :, :, :, :]
    gt_reconstruction = gt_reconstruction[idx, :, :, :]
    shadow_mask = shadow_mask[idx, :, :, :]
    light_mask = light_mask[idx, :, :, :]
    occlusion_mask = occlusion_mask[idx, :, :, :]
    occlusion_rgb = occlusion_rgb[idx, :, :, :, :]

    columns = [
        "org_img",
        "org_occlusion",
        "gt_reconstruction",
        "shadow_mask",
        "light_mask",
        "occlusion_mask",
        "occlusion_rgb",
        "complete_reconstruction",
    ]

    occlusion_rec = torch.where(
        occlusion_mask.unsqueeze(0).repeat(3, 1, 1, 1) < 0.5,
        (
            gt_reconstruction.unsqueeze(1).repeat(1, 10, 1, 1)
            * shadow_mask.unsqueeze(0).repeat(3, 1, 1, 1)
            + light_mask.unsqueeze(0).repeat(3, 1, 1, 1)
        ),
        occlusion_rgb,
    )

    my_data = [
        [
            wandb.Image(ToPILImage(y), caption=columns[0]),
            [
                wandb.Image(ToPILImage(x[:, img, :, :]), caption=columns[1])
                for img in range(x.shape[1])
            ],
            wandb.Image(ToPILImage(gt_reconstruction), caption=columns[2]),
            [
                wandb.Image(ToPILImage(shadow_mask[img, :, :]), caption=columns[3])
                for img in range(shadow_mask.shape[0])
            ],
            [
                wandb.Image(ToPILImage(light_mask[img, :, :]), caption=columns[4])
                for img in range(light_mask.shape[0])
            ],
            [
                wandb.Image(ToPILImage(occlusion_mask[img, :, :]), caption=columns[5])
                for img in range(occlusion_mask.shape[0])
            ],
            [
                wandb.Image(ToPILImage(occlusion_rgb[:, img, :, :]), caption=columns[6])
                for img in range(occlusion_rgb.shape[1])
            ],
            [
                wandb.Image(ToPILImage(occlusion_rec[:, img, :, :]), caption=columns[7])
                for img in range(occlusion_rec.shape[1])
            ],
        ]
    ]

    logger.log_table(key="input_output", columns=columns, data=my_data)


def pre_train_log_images(logger, gt_reconstruction, x):
    idx = torch.randint(0, x.shape[0], (1,)).item()

    x = x[idx, :, :, :, :]
    gt_reconstruction = gt_reconstruction[idx, :, :, :]

    columns = ["input", "gt_reconstruction"]

    my_data = [
        [
            [
                wandb.Image(ToPILImage(x[:, img, :, :]), caption=columns[0])
                for img in range(x.shape[1])
            ],
            [
                wandb.Image(
                    ToPILImage(gt_reconstruction[:, img, :, :]), caption=columns[1]
                )
                for img in range(gt_reconstruction.shape[1])
            ],
        ]
    ]

    logger.log_table(key="input_output", columns=columns, data=my_data)
