import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf

# from src.model import Model
from utils.parser import parse_arguments
from models.decomposer import Decomposer
from data.siar_data import SIARDataModule


def main(args):
    config = OmegaConf.load(args.config)
    wandb_logger = WandbLogger(config=config, project="HTCV")

    siar = SIARDataModule(config.data.dir, config.train.batch_size)
    siar.setup("train", config.train.debug)

    model = Decomposer(config=config.model)
    trainer = pl.Trainer(
        max_epochs=config.train.max_epochs,
        logger=wandb_logger,
        default_root_dir="checkpoints",
        accelerator="gpu",
    )

    trainer.fit(
        model,
        datamodule=siar,
    )

    siar.setup("test", config.train.debug)
    trainer.test(model, datamodule=siar)

    return


if __name__ == "__main__":
    args = parse_arguments()

    main(args)
