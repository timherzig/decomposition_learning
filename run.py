import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf

# from src.model import Model
from utils.parser import parse_arguments
from models.decomposer import Decomposer
from data.siar_data import SIARDataModule

import subprocess


def get_git_commit():
    process = subprocess.Popen(
        ["git", "rev-parse", "--short", "HEAD"], shell=False, stdout=subprocess.PIPE
    )
    git_head_hash = process.communicate()[0].strip()

    return git_head_hash


def main(args):
    config = OmegaConf.load(args.config)
    wandb_logger = WandbLogger(config=config, project="HTCV")

    siar = SIARDataModule(config.data.dir, config.train.batch_size)
    siar.setup("train", config.train.debug)

    model = Decomposer(config=config.model)
    trainer = pl.Trainer(
        max_epochs=config.train.max_epochs,
        logger=wandb_logger,
        default_root_dir=f"checkpoints",
        log_every_n_steps=config.train.log_every_n_steps,
        accelerator="gpu",
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
