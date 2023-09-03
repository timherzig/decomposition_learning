import os

from omegaconf import OmegaConf

from utils.utils import setup_evaluation, parse_arguments, get_git_commit


def main(args):
    config, siar, model, trainer = setup_evaluation(args)

    siar.setup("test")
    trainer.test(model, datamodule=siar)

    return


if __name__ == "__main__":
    args = parse_arguments()

    main(args)
