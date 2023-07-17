import os
import torch

import pandas as pd

from PIL import Image
from copy import deepcopy
from lightning.pytorch import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor

from src.models.utils.preprocessing import get_shadow_light_gt, get_occlusion_gt
from src.models.utils.utils import get_class


class SIAR(Dataset):
    """
    PyTorch Dataset for the provided dataset
    """

    def __init__(
        self,
        split: str,
        split_version: str = "",
        sanity_check=False,
        manual_dataset_path=None,
    ) -> None:
        super().__init__()
        """SIAR Dataset

        Args:
            split (str): train, val or test
            split_version (str, optional): Which split to use. Defaults to "split-1_80_10_10".
            sanity_check (bool, optional): Whether to use only one entry and overfit to it. Defaults to False.
            manual_dataset_path ([type], optional): Path to dataset. Defaults to None.
        """
        if manual_dataset_path:
            self.path_to_dataset = manual_dataset_path
        else:
            self.path_to_dataset = os.path.join(os.getcwd(), "data/SIAR")

        if split_version == "":
            path_to_split = os.path.join(self.path_to_dataset, f"{split}.csv")
        else:
            path_to_split = os.path.join(
                f"{self.path_to_dataset}/..",
                "data_splits",
                split_version,
                f"{split}.csv",
            )

        # Load ids of current split
        self.df = pd.read_csv(path_to_split, index_col=None)
        # Add directory paths to those ids
        self.df["dir"] = [
            os.path.join(self.path_to_dataset, str(x)) for x in self.df["id"]
        ]

        # For sanity check, we only want to use one entry
        # and want to completely overfit to it
        if sanity_check:
            self.df = self.df.sample(frac=1).reset_index(drop=True)
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
        sl = torch.zeros_like(images)
        occ = torch.zeros_like(images)

        return (images, ground_truth, sl, occ)

    def __len__(self):
        return len(self.df)


class SIAR_SL(SIAR):
    """
    SIAR Dataset with shadow and light approximation targets.

    Returns:
        images (torch.Tensor): [C, N, H, W] tensor of images. \\
        ground_truth (torch.Tensor): [C, H, W] tensor of ground truth image. \\
        sl (torch.Tensor): [C, N, H, W] tensor of shadow and light approximation. Generated on the fly. \\
        occ (torch.Tensor): [C, N, H, W] tensor of occlusion approximation. In this case we return zeros, as we don't need it in the SL dataset.

    C is number of channels, N is number of images should be 10, H and W are height and width of images.
    """

    def __init__(
        self,
        split: str,
        split_version: str = "",
        sanity_check=False,
        manual_dataset_path=None,
    ) -> None:
        """
        Args:
            split (str): train, val or test
            split_version (str, optional): Which split to use. Defaults to "split-1_80_10_10".
            sanity_check (bool, optional): Whether to use only one entry and overfit to it. Defaults to False.
            manual_dataset_path (str, optional): Path to dataset. Defaults to None.
        """
        super().__init__(split, split_version, sanity_check, manual_dataset_path)

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

        sl = get_shadow_light_gt(ground_truth, images)
        occ = torch.zeros_like(sl)

        return (images, ground_truth, sl, occ)


class SIAR_OCC(SIAR):
    """
    SIAR Dataset with occlusion approximation targets.

    Returns:
        images (torch.Tensor): [C, N, H, W] tensor of images.
        ground_truth (torch.Tensor): [C, H, W] tensor of ground truth image.
        sl (torch.Tensor): [C, N, H, W] tensor of shadow and light approximation. In this case we return zeros, as we don't need it in the OCC dataset.
        occ (torch.Tensor): [C, N, H, W] tensor of occlusion approximation. Loaded from pre-generated files.

    C is number of channels, N is number of images should be 10, H and W are height and width of images.
    """

    def __init__(
        self,
        split: str,
        split_version: str = "",
        sanity_check=False,
        manual_dataset_path=None,
    ) -> None:
        """
        Args:
            split (str): train, val or test
            split_version (str, optional): Which split to use. Defaults to "split-1_80_10_10".
            sanity_check (bool, optional): Whether to use only one entry and overfit to it. Defaults to False.
            manual_dataset_path (str, optional): Path to dataset. Defaults to None.
        """
        super().__init__(split, split_version, sanity_check, manual_dataset_path)

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

        sample = dir.split("/")[-1]
        path_to_occ = f"/srv/data/SOLP/nextBillion/SIAR_OCC/SIAR_OCC/{sample}.pt"
        occ = torch.load(path_to_occ)
        sl = torch.zeros_like(images)

        return (images, ground_truth, sl, occ)


class SIAR_OCC_GENERATION(SIAR):
    """
    SIAR Dataset with occlusion approximation targets.

    Returns:
        images (torch.Tensor): [C, N, H, W] tensor of images.
        ground_truth (torch.Tensor): [C, H, W] tensor of ground truth image.
        sl (torch.Tensor): [C, N, H, W] tensor of shadow and light approximation. In this case we return zeros, as we don't need it in the OCC dataset.
        occ (torch.Tensor): [C, N, H, W] tensor of occlusion approximation. Loaded from pre-generated files.

    C is number of channels, N is number of images should be 10, H and W are height and width of images.
    """

    def __init__(
        self,
        split: str,
        split_version: str = "",
        sanity_check=False,
        manual_dataset_path=None,
    ) -> None:
        """
        Args:
            split (str): train, val or test
            split_version (str, optional): Which split to use. Defaults to "split-1_80_10_10".
            sanity_check (bool, optional): Whether to use only one entry and overfit to it. Defaults to False.
            manual_dataset_path (str, optional): Path to dataset. Defaults to None.
        """
        super().__init__(split, split_version, sanity_check, manual_dataset_path)

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

        occ = get_occlusion_gt(ground_truth, images)
        sample = dir.split("/")[-1]
        path_to_occ = os.path.join(self.path_to_dataset, f"../SIAR_OCC/{sample}.pt")
        occ = torch.load(path_to_occ)
        sl = torch.zeros_like(images)

        return (images, ground_truth, sl, occ)


class SIARDataModule(LightningDataModule):
    """
    Data Module for SIAR Dataset
    """

    def __init__(
        self,
        batch_size: int,
        dataset,
        split_dir: str = "split-1_80_10_10",
        num_workers: int = 0,
        manual_dataset_path=None,
    ) -> None:
        super().__init__()

        self.manual_dataset_path = manual_dataset_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split_dir = split_dir
        self.dataset = get_class(dataset, ["src.data.siar_data"])

    def prepare_data(self) -> None:
        return

    def setup(self, stage: str, sanity_check=False) -> None:
        if stage == "test":
            self.siar_test = self.dataset(
                split="test",
                sanity_check=sanity_check,
                manual_dataset_path=self.manual_dataset_path,
            )
        elif stage == "train":
            self.siar_train = self.dataset(
                split="train",
                sanity_check=sanity_check,
                manual_dataset_path=self.manual_dataset_path,
            )
            # If we are doing a sanity check, we want to use the same data for validation and overfit to it
            self.siar_val = (
                deepcopy(self.siar_train)
                if sanity_check
                else self.dataset(
                    split="val",
                    sanity_check=sanity_check,
                    manual_dataset_path=self.manual_dataset_path,
                )
            )

    def train_dataloader(self):
        return DataLoader(
            self.siar_train,
            batch_size=self.batch_size,
            # num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.siar_test,
            batch_size=self.batch_size,
        )  # num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(
            self.siar_val,
            batch_size=self.batch_size,
        )  # num_workers=self.num_workers)
