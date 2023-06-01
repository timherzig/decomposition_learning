import os
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf

# from src.model import Model
from utils.parser import parse_arguments
from models.decomposer import Decomposer
from data.siar_data import SIARDataModule
from utils.git_commit import get_git_commit


def main(args):
    config = OmegaConf.load(args.config)
    if not config.train.debug:
        wandb_logger = WandbLogger(config=config, project="HTCV")

    siar = SIARDataModule(config.data.dir, config.train.batch_size)
    siar.setup("train", config.train.debug)

    model = (
        Decomposer(
            config=config,
            log_dir=(wandb_logger.experiment.name if config.train.pre_train else None),
        )
        if not config.model.checkpoint
        else Decomposer.load_from_checkpoint(
            config.model.checkpoint,
            config=config,
            log_dir=(wandb_logger.experiment.name if config.train.pre_train else None),
        )
    )

    trainer = pl.Trainer(
        max_epochs=config.train.max_epochs,
        logger=wandb_logger if not config.train.debug else None,
        default_root_dir=f"checkpoints",
        log_every_n_steps=config.train.log_every_n_steps
        if not config.train.debug
        else None,
        accelerator=config.train.device,
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
