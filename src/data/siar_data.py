import os
import torch

import pandas as pd

from PIL import Image
from copy import deepcopy
from lightning.pytorch import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor

from src.models.utils.preprocessing import get_shadow_light_gt


class SIAR(Dataset):
    """
    PyTorch Dataset for the provided dataset
    """

    def __init__(
        self,
        split: str,
        split_version: str = "split-1_80_10_10",
        sanity_check=False,
        manual_dataset_path=None,
    ) -> None:
        super().__init__()
        """SIAR Dataset

        Args:
            split (str): train, val or test
            split_version (str, optional): Which split to use. Defaults to "split-1_80_10_10".
            sanity_check (bool, optional): Whether to use only one entry and overfit to it. Defaults to False.
        """

        if manual_dataset_path:
            path_to_dataset = manual_dataset_path
        else:
            path_to_dataset = os.path.join(os.getcwd(), "data/SIAR")
        path_to_split = os.path.join(
            os.getcwd(), "data/data_splits", split_version, f"{split}.csv"
        )

        # Load ids of current split
        self.df = pd.read_csv(path_to_split, index_col=None)
        # Add directory paths to those ids
        self.df["dir"] = [os.path.join(path_to_dataset, str(x)) for x in self.df["id"]]

        # For sanity check, we only want to use one entry
        # and want to completely overfit to it
        if sanity_check:
            self.df = self.df[:1]
            self.df = pd.concat(
                [self.df, self.df], ignore_index=True
            )  # Now we can batches but they will all be the same entry

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

        return images, ground_truth, get_shadow_light_gt(ground_truth, images)

    def __len__(self):
        return len(self.df)


class SIARDataModule(LightningDataModule):
    """
    Data Module for SIAR Dataset
    """

    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        split_dir: str = "split-1_80_10_10",
        manual_dataset_path=None,
    ) -> None:
        super().__init__()

        self.manual_dataset_path = manual_dataset_path
        self.batch_size = batch_size
        self.split_dir = split_dir
        self.num_workers = num_workers

    def prepare_data(self) -> None:
        return

    def setup(self, stage: str, sanity_check=False) -> None:
        if stage == "test":
            self.siar_test = SIAR(
                split="test",
                sanity_check=sanity_check,
                manual_dataset_path=self.manual_dataset_path,
            )
        elif stage == "train":
            self.siar_train = SIAR(
                split="train",
                sanity_check=sanity_check,
                manual_dataset_path=self.manual_dataset_path,
            )
            # If we are doing a sanity check, we want to use the same data for validation and overfit to it
            self.siar_val = (
                deepcopy(self.siar_train)
                if sanity_check
                else SIAR(
                    split="val",
                    sanity_check=sanity_check,
                    manual_dataset_path=self.manual_dataset_path,
                )
            )

    def train_dataloader(self):
        return DataLoader(
            self.siar_train, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.siar_test, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.siar_val, batch_size=self.batch_size, num_workers=self.num_workers
        )
