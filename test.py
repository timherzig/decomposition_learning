import torch
from data.dataloader import DecompositionDataModule
from models.decomposer import Decomposer
import lightning.pytorch as pl

from omegaconf import OmegaConf

from utils.parser import parse_arguments


def main(args):
    # config = OmegaConf.load(args.config)

    # siar = SIARDataModule(config.data.dir, 4)
    # siar.setup('train')

    # print(siar.siar.df)
    config = OmegaConf.load(args.config)
    dataModule = DecompositionDataModule(datapath="data/SIAR", batch_size=2)
    dataModule.setup(stage='fit')
    model =  Decomposer(config.unet)
    trainer = pl.Trainer(max_epochs=1)
    trainer.fit(model, train_dataloaders=dataModule.train_dataloader())
    return

if __name__ == '__main__':
    args = parse_arguments()
    
    main(args)