import os
import subprocess
from argparse import ArgumentParser

import lightning.pytorch as pl
from omegaconf import OmegaConf
from lightning.pytorch.loggers import WandbLogger

from src.models.model import Decomposer
from src.data.siar_data import SIARDataModule

from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import torch


def setup_training(args):
    config = OmegaConf.load(args.config)
    print("-----------------")
    print(f"Config: {args.config}")
    print("-----------------")

    if not config.train.debug:
        config_name = os.path.basename(args.config).split(".")[0]
        wandb_logger = WandbLogger(config=config, project="HTCV", name=config_name)
        print(f"Experiment name: {wandb_logger._name}")
        print("-----------------")

    log_dir = None
    if config.train.pre_train:
        if not config.train.debug:
            log_dir = f"swin_checkpoints/{wandb_logger._name}"
        else:
            log_dir = "swin_checkpoints/debug"
        os.makedirs(log_dir, exist_ok=True)

    if config.train.stage in ["train_gt", "train_sl", "train_ob"]:
        if not config.train.debug:
            log_dir = f"decoder_checkpoints/{wandb_logger._name}"
        else:
            log_dir = (
                f"decoder_checkpoints/{args.config.split('/')[-1].split('.')[0]}_debug"
            )

        os.makedirs(log_dir, exist_ok=True)

    siar = SIARDataModule(
        config.train.batch_size,
        config.data.dataset,
        config.data.split_dir,
        args.num_workers,
        args.data_dir,
    )

    if not config.model.checkpoint:
        model = Decomposer(
            config=config,
            log_dir=log_dir,
        )
    else:
        model = Decomposer.load_from_checkpoint(
            config.model.checkpoint, config=config, log_dir=log_dir
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
        # accelerator="cpu",
        strategy=config.train.strategy,
        accumulate_grad_batches=config.train.accumulate_grad_batches,
    )

    print("Setup complete")
    print("-----------------")

    return config, siar, model, trainer


def setup_evaluation(args):
    config = OmegaConf.load(args.config)
    print("-----------------")
    print(f"Config: {args.config}")
    print("-----------------")

    assert config.model.checkpoint, "No checkpoint provided for evaluation"
    print("CHECKPOINT: ", config.model.checkpoint)

    siar = SIARDataModule(
        config.train.batch_size,
        config.data.dataset,
        config.data.split_dir,
        config.data.num_workers,
        args.data_dir,
    )

    if config.train.device == "cpu":
        map_location = torch.device("cpu")
    else:
        map_location = None

    model = Decomposer.load_from_checkpoint(
        config.model.checkpoint,
        config=config,
        log_dir=None,
        eval_output=config.data.eval_output,
        map_location=map_location,
    )

    trainer = pl.Trainer(
        max_epochs=config.train.max_epochs,
        logger=None,
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
    parser.add_argument("--num-workers", type=int, default=8)

    return parser.parse_args()


def get_git_commit():
    process = subprocess.Popen(
        ["git", "rev-parse", "--short", "HEAD"], shell=False, stdout=subprocess.PIPE
    )
    return process.communicate()[0].strip()
