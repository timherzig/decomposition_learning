from argparse import ArgumentParser


def parse_arguments():
    parser = ArgumentParser()

    # parser.add_argument("--config", type=str, help="", default="config/default.yaml")
    parser.add_argument(
        "--config", type=str, help="", default="config/swin_unet_unet_unet.yaml"
    )

    parser.add_argument("--data", help="", type=str)

    return parser.parse_args()
