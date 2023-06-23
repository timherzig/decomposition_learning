import os

from omegaconf import OmegaConf

from utils.utils import setup_training, parse_arguments, get_git_commit


import os
import subprocess
from argparse import ArgumentParser

import lightning.pytorch as pl
from omegaconf import OmegaConf
from lightning.pytorch.loggers import WandbLogger

from src.models.model import Decomposer
from src.data.siar_data import SIARDataModule


def main(args):
    # config, siar, model, trainer = setup_training(args)
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
    siar.setup("train", config.data.sanity_check)

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

    trainer.fit(model, datamodule=siar)

    siar.setup("test", config.train.debug)
    trainer.test(model, datamodule=siar)

    conf = OmegaConf.merge([config, OmegaConf.create({"git_commit": get_git_commit()})])

    yaml_data: str = OmegaConf.to_yaml(conf)

    with open(os.path.join(trainer.log_dir, "config.yaml"), "w") as f:
        OmegaConf.save(conf, f)

    return


if __name__ == "__main__":
    args = parse_arguments()

    main(args)
