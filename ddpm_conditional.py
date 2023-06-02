import os
import copy
import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
from tqdm import tqdm
from torch import optim
from modules import UNet_conditional, EMA
import logging
from omegaconf import OmegaConf
import wandb

from data.siar_data import SIARDataModule
from utils.parser import parse_arguments


logging.basicConfig(
    format="%(asctime)s - %(levelname)s: %(message)s",
    level=logging.INFO,
    datefmt="%I:%M:%S",
)


def to_pil(x):
    return ToPILImage()(x)


def format_table_logging(images, outputs, target):
    # images --- [3, 10, H, W]
    # outputs --- [n, 3, H, W]
    # target --- [3, H, W]
    columns = ["input", "output", "target"]
    my_data = [
        [
            [
                wandb.Image(to_pil(images[:, idx, :, :]), caption=columns[0])
                for idx in range(images.shape[1])
            ],
            [
                wandb.Image(
                    to_pil(img),
                    caption=columns[1],
                )
                for img in outputs
            ],
            wandb.Image(
                to_pil(target),
                caption=columns[2],
            ),
        ]
    ]

    return wandb.Table(columns=columns, data=my_data)


class Diffusion:
    def __init__(
        self,
        noise_steps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        img_size=256,
        device="cuda",
    ):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.img_size = img_size
        self.device = device

        self.to_pil = ToPILImage()

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[
            :, None, None, None
        ]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n, conditioning, cfg_scale=3):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            # copy conditioning 4 times
            conditioning = torch.unsqueeze(conditioning, 0)
            conditioning = torch.cat([conditioning] * n, dim=0)
            conditioning = conditioning.to(self.device)  # --- [n, 3, 10, 256, 256]
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t, conditioning)
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    predicted_noise = torch.lerp(
                        uncond_predicted_noise, predicted_noise, cfg_scale
                    )
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = (
                    1
                    / torch.sqrt(alpha)
                    * (
                        x
                        - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise
                    )
                    + torch.sqrt(beta) * noise
                )
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x


def train(args):
    config = OmegaConf.load(args.config)
    if config.train.device == "None":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.train.device

    if not config.train.debug:
        wandb_logger = wandb.init(config=config, project="HTCV")
    else:
        wandb_logger = None

    dataloader = SIARDataModule(config.data.dir, config.train.batch_size)
    dataloader.setup(stage="train")
    train_dataloader = dataloader.train_dataloader()
    val_dataloader = dataloader.val_dataloader()
    model = UNet_conditional(
        device=device,
        img_size=config.data.img_size,
        decomposer_config=config.decomposer,
    ).to(device)

    print("Parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    optimizer = optim.AdamW(model.parameters(), lr=config.train.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=config.data.img_size, device=device)

    # get len of lightning datamodule
    l = len(train_dataloader)
    ema = EMA(0.995)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    global_step = 0

    for epoch in range(config.train.max_epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(train_dataloader)
        for i, (images, gt) in enumerate(pbar):
            images = images.to(device)  # --- [B, 3, 10, 256, 256]
            gt = gt.to(device)  # --- [B, 3, 256, 256]
            t = diffusion.sample_timesteps(images.shape[0]).to(device)

            # Add noise to ground truth reconstruction
            x_t, noise = diffusion.noise_images(gt, t)  # --- [B, 3, 256, 256]

            # Add images as conditioning to model (will be passed through decomposer)
            predicted_noise = model(x_t, t, images)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.step_ema(ema_model, model)

            pbar.set_postfix(MSE=loss.item())

            # Log to wandb
            if not config.train.debug and global_step % config.train.log_every == 0:
                wandb_logger.log(
                    {
                        "train_loss": loss.item(),
                    },
                    step=global_step,
                )

            global_step += 1

        # Evaluate model
        if epoch % config.train.eval_every == 0 and epoch > 0:
            with torch.no_grad():
                losses = []  # --- list of mean losses for each batch
                for images, gts in val_dataloader:
                    # images --- [B, 3, 10, 256, 256]
                    # gt --- [B, 3, 256, 256]
                    for batch_elem, gt in zip(images, gts):
                        predictions = diffusion.sample(ema_model, 4, batch_elem).to(
                            device
                        )  # --- already takes care of setting to eval and train
                        batch_elem_losses = torch.tensor(
                            [mse(pred, gt.to(device)) for pred in predictions]
                        )
                        mean_loss = torch.mean(batch_elem_losses)
                        losses.append(mean_loss.item())
                        # if config.train.debug:
                        # break
                    # if config.train.debug:
                    # break
                # Get mean loss over all batches
                mean_loss = torch.mean(torch.tensor(losses))

            logging.info(f"Epoch {epoch}: Val MSE: {mean_loss.item()}")

            # Log to wandb
            if not config.train.debug:
                wandb_logger.log(
                    {
                        "val_loss": mean_loss.item(),
                    },
                    step=global_step,
                )
                table_to_log = format_table_logging(batch_elem, predictions, gt)
                wandb_logger.log({"input_output_diffusion": table_to_log})

        # Save model
        if (
            not config.train.debug
            and epoch % config.train.save_every == 0
            and epoch > 0
        ):
            # if dir does not exist, create it
            if not os.path.exists(wandb_logger.run.dir):
                os.makedirs(wandb_logger.run.dir)

            torch.save(
                model.state_dict(),
                os.path.join(wandb_logger.run.dir, f"model_{epoch}.pt"),
            )


def main():
    args = parse_arguments()
    train(args)


if __name__ == "__main__":
    main()
