import os
import torch
import subprocess
from argparse import ArgumentParser

import lightning.pytorch as pl
from omegaconf import OmegaConf
from lightning.pytorch.loggers import WandbLogger

from src.models.model import Decomposer
from src.data.siar_data import SIARDataModule


def setup_training(args):
    config = OmegaConf.load(args.config)
    print("-----------------")
    print(f"Config: {args.config}")
    print("-----------------")

    if not config.train.debug:
        config_name = os.path.basename(args.config).split(".")[0]
        wandb_logger = WandbLogger(config=config, project="HTCV", name=config_name)
        print(f"Experiment name: {wandb_logger.experiment.name}")
        print("-----------------")

    if config.train.pre_train:
        if not config.train.debug:
            log_dir = f"swin_checkpoints/{wandb_logger.experiment.name}"
        else:
            log_dir = "swin_checkpoints/debug"
        os.makedirs(log_dir, exist_ok=True)

    siar = SIARDataModule(
        config.train.batch_size,
        config.data.split_dir,
        args.data_dir,
    )

    if not config.model.checkpoint:
        model = Decomposer(
            config=config,
            log_dir=log_dir if config.train.pre_train else None,
        )
    else:
        model = Decomposer.load_from_checkpoint(
            config.model.checkpoint,
            config=config,
            log_dir=log_dir if config.train.pre_train else None,
        )

    if not config.train.debug:
        wandb_logger.watch(model, log="all", log_freq=100)

    trainer = pl.Trainer(
        max_epochs=config.train.max_epochs,
        logger=None if config.train.debug else wandb_logger,
        default_root_dir="checkpoints",
        log_every_n_steps=None
        if config.train.debug
        else config.train.log_every_n_steps,
        accelerator=config.train.device,
        strategy=config.train.strategy,
        accumulate_grad_batches=config.train.accumulate_grad_batches,
    )

    print("Model loaded")
    print("-----------------")

    return config, siar, model, trainer


def parse_arguments():
    parser = ArgumentParser()

    # parser.add_argument("--config", type=str, help="", default="config/default.yaml")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config.yaml file.",
        default="config/default.yaml",
    )

    parser.add_argument("--data-dir", help="Path to dataset", type=str, default=None)

    return parser.parse_args()


def get_git_commit():
    process = subprocess.Popen(
        ["git", "rev-parse", "--short", "HEAD"], shell=False, stdout=subprocess.PIPE
    )
    return process.communicate()[0].strip()
