import lightning.pytorch as pl
from omegaconf import OmegaConf

# from src.model import Model
from utils.parser import parse_arguments
from models.decomposer import Decomposer
from data.siar_data import SIARDataModule

def main(args):
    config = OmegaConf.load(args.config)

    siar = SIARDataModule(config.data.dir, 2)
    siar.setup('train')

    model = Decomposer(unet_config=config.unet)
    trainer = pl.Trainer(max_epochs=1)
    trainer.fit(model, train_dataloaders=siar.train_dataloader())

    return

if __name__ == '__main__':
    args = parse_arguments()
    
    main(args)