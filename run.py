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

    siar = SIARDataModule(config.data.dir, 2)
    siar.setup("train")

    model = Decomposer(config=config.model)
    trainer = pl.Trainer(max_epochs=1, logger=wandb_logger)
    trainer.fit(
        model,
        train_dataloaders=siar.train_dataloader(),
        val_dataloaders=siar.val_dataloader(),
    )

    return


if __name__ == "__main__":
    args = parse_arguments()

    main(args)
