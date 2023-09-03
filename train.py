import os

from omegaconf import OmegaConf

from utils.utils import setup_training, parse_arguments, get_git_commit


def main(args):
    config, siar, model, trainer = setup_training(args)

    siar.setup("train")
    trainer.fit(model, datamodule=siar)
    return


if __name__ == "__main__":
    args = parse_arguments()

    main(args)
