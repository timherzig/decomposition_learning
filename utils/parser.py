from argparse import ArgumentParser


def parse_arguments():
    parser = ArgumentParser()

    parser.add_argument("--config", type=str, help="", default="configs/default.yaml")

    return parser.parse_args("")
