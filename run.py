from omegaconf import OmegaConf

from data.siar_data import SIARDataModule

from utils.parser import parse_arguments

def main(args):
    config = OmegaConf.load(args.config)

    siar = SIARDataModule(config.data.dir, 4)
    siar.setup('train')

    print(siar.siar_train.df)

    return

if __name__ == '__main__':
    args = parse_arguments()
    
    main(args)