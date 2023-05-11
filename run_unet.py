import torch
from data.siar_data import SIARDataModule
from models.decomposer import Decomposer
import lightning.pytorch as pl

from omegaconf import OmegaConf

from utils.parser import parse_arguments

'''
Script to run swin transformer with 3DUnet upsampling method. Use config/unet.yaml.
'''

def main(args):

    config = OmegaConf.load(args.config)
    dataModule = SIARDataModule(config.data.dir, batch_size=2)
    dataModule.setup(stage='train')
    model =  Decomposer(swin_config=config.model.swin, up_sampling="unet", unet_config=config.unet)
    trainer = pl.Trainer(max_epochs=1)
    trainer.fit(model, train_dataloaders=dataModule.train_dataloader())
    return

if __name__ == '__main__':
    args = parse_arguments()
    
    main(args)