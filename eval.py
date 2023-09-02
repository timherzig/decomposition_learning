import os

from omegaconf import OmegaConf

from utils.utils import setup_evaluation, parse_arguments, get_git_commit


def main(args):
    config, siar, model, trainer = setup_evaluation(args)



    siar.setup("test", config.train.debug)
    trainer.test(model, datamodule=siar)

    conf = OmegaConf.merge([config, OmegaConf.create({"git_commit": 'test'})])

    with open(os.path.join(trainer.log_dir, "config.yaml"), "w") as f:
        OmegaConf.save(conf, f)

    return


if __name__ == "__main__":
    args = parse_arguments()

    main(args)
