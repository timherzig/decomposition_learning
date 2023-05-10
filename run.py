from omegaconf import OmegaConf

# from src.model import Model
from data.siar_data import SIARDataModule

from models.decomposer import Decomposer

from utils.parser import parse_arguments

def main(args):
    config = OmegaConf.load(args.config)

    siar = SIARDataModule(config.data.dir, 4)
    model = Decomposer(unet_config=config.unet)
    siar.setup('val')
    


    return

if __name__ == '__main__':
    args = parse_arguments()
    
    main(args)