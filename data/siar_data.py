import os
import torch

import pandas as pd

from PIL import Image
from lightning.pytorch import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor


class SIAR(Dataset):
    """
    PyTorch Dataset for the provided dataset
    """

    def __init__(self, data_dir: str, split: str, debug) -> None:
        super().__init__()

        self.df = pd.read_csv(os.path.join(data_dir, split + ".csv"), index_col=None)
        self.df["dir"] = [os.path.join(data_dir, str(x)) for x in self.df["id"]]

        if debug:
            self.df = self.df[:1]

    def __getitem__(self, index):
        dir = self.df.iloc[index]["dir"]

        ground_truth = os.path.join(dir, "gt.png")
        assert os.path.exists(ground_truth) == True, f"{ground_truth} does not exist"
        ground_truth = ToTensor()(Image.open(ground_truth))

        images = torch.stack(
            [
                ToTensor()(Image.open(os.path.join(dir, x)))
                for x in os.listdir(dir)
                if x.split(".")[0].isnumeric()
            ]
        )
        images = torch.swapaxes(images, 0, 1)

        return images, ground_truth

    def __len__(self):
        return len(self.df)


class SIARDataModule(LightningDataModule):
    """
    Data Module for the
    """

    def __init__(self, data_dir: str, batch_size: int) -> None:
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size

    def prepare_data(self) -> None:
        return

    def setup(self, stage: str, debug=False) -> None:
        if stage == "train":
            self.siar_train = SIAR(self.data_dir, "train", debug)
            if debug:
                self.siar_val = self.siar_train.copy()
            self.siar_val = SIAR(self.data_dir, "val", debug)
        if stage == "test":
            self.siar_test = SIAR(self.data_dir, "test", debug)

    def train_dataloader(self):
        return DataLoader(self.siar_train, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.siar_test, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.siar_val, batch_size=self.batch_size)
