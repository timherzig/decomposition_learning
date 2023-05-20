import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf

# from src.model import Model
from utils.parser import parse_arguments
from models.decomposer import Decomposer
from data.siar_data import SIARDataModule


def main(args):
    config = OmegaConf.load(args.config)

    siar = SIARDataModule(config.data.dir, config.train.batch_size)
    siar.setup("train", config.train.debug)

    model = Decomposer(config=config.model)

    if config.wandb.logger:
        wandb_logger = WandbLogger(config=config, project="HTCV")

        trainer = pl.Trainer(
            max_epochs=config.train.max_epochs,
            logger=wandb_logger,
            default_root_dir="checkpoints",
            log_every_n_steps=config.train.log_every_n_steps
        )
    else:
        trainer = pl.Trainer(
            max_epochs=config.train.max_epochs,
            default_root_dir="checkpoints",
            log_every_n_steps=config.train.log_every_n_steps)
        
    trainer.fit(
        model,
        train_dataloaders=siar.train_dataloader(),
        val_dataloaders=siar.val_dataloader(),
    )

    # siar.setup("test", config.train.debug)
    # trainer.test(model, test_dataloaders=siar.test_dataloader())

    return


if __name__ == "__main__":
    args = parse_arguments()

    main(args)
