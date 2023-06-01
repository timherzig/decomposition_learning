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
    device = config.train.device

    wandb_logger = wandb.init(config=config, project="HTCV")

    dataloader = SIARDataModule(config.data.dir, config.train.batch_size)
    dataloader.setup(stage="train")
    train_dataloader = dataloader.train_dataloader()
    val_dataloader = dataloader.val_dataloader()
    model = UNet_conditional(device=device, decomposer_config=config.decomposer).to(
        device
    )

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
            if global_step % config.train.log_every == 0:
                wandb_logger.log(
                    {
                        "loss": loss.item(),
                        "global_step": global_step,
                    },
                    step=global_step,
                )

            global_step += 1

        # Evaluate model
        if epoch % config.train.eval_every == 0:
            with torch.no_grad():
                losses = []
                for i, (images, gt) in enumerate(val_dataloader):
                    xs = diffusion.sample(ema_model, 4, images).to(
                        device
                    )  # --- already takes care of setting to eval and train
                    mean_loss = torch.mean([mse(x, gt.to(device)) for x in xs])
                    losses.append(mean_loss.item())
                mean_loss = torch.mean(torch.tensor(losses))

            logging.info(f"Epoch {epoch}: Val MSE: {mean_loss.item()}")

            # Log to wandb
            wandb_logger.log(
                {
                    "val_mse": mean_loss.item(),
                },
                step=global_step,
            )
            imgs = make_grid(xs, nrow=2)  # --- Only log the last batch
            imgs = wandb.Image(imgs)
            wandb_logger.log({"val_samples": imgs}, step=global_step)

        # Save model
        if epoch % config.train.save_every == 0:
            torch.save(
                model.state_dict(),
                os.path.join(wandb_logger.run.dir, f"model_{epoch}.pt"),
            )


def main():
    args = parse_arguments()
    train(args)


if __name__ == "__main__":
    main()
