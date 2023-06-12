from argparse import ArgumentParser


def parse_arguments():
    parser = ArgumentParser()

    parser.add_argument("--config", type=str, help="", required=True)

    return parser.parse_args("")
