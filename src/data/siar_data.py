import os
import torch

import pandas as pd
import numpy as np

from PIL import Image
from copy import deepcopy
from lightning.pytorch import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor

from src.models.utils.preprocessing import get_shadow_light_gt, get_occlusion_gt
from src.models.utils.utils import get_class, images_to_tensor


class SIAR(Dataset):
    """
    PyTorch Dataset for the provided dataset
    """

    def __init__(
        self,
        split: str,
        split_version: str = "split-1_80_10_10",
        manual_dataset_path=None,
    ) -> None:
        super().__init__()
        """SIAR Dataset

        Args:
            split (str): train, val or test
            split_version (str, optional): Which split to use. Defaults to "split-1_80_10_10".
            manual_dataset_path ([type], optional): Path to dataset. Defaults to None.
        """
        if manual_dataset_path:
            self.path_to_dataset = manual_dataset_path
        else:
            self.path_to_dataset = os.path.join(os.getcwd(), "data")

        path_to_split = os.path.join(
            f"{self.path_to_dataset}",
            "data_splits",
            split_version,
            f"{split}.csv",
        )

        # Load ids of current split
        self.df = pd.read_csv(path_to_split, index_col=None)
        # Add directory paths to those ids
        self.df["dir"] = [
            os.path.join(self.path_to_dataset, "SIAR", str(x)) for x in self.df["id"]
        ]

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


class SIAR_EVAL(Dataset):
    """
    SIAR Dataset desigend for inference on the evaluation dataset (Not the same dataset as initial SIAR!).

    Returns:
        images (torch.Tensor): [C, N, H, W] tensor of images. \\
        ground_truth (torch.Tensor): [C, H, W] tensor of ground truth image. \\
        sl (torch.Tensor): [C, N, H, W] tensor of shadow and light approximation. Generated on the fly. \\
        occ (torch.Tensor): [C, N, H, W] tensor of occlusion approximation. In this case we return zeros, as we don't need it in the SL dataset.
        dir (string): Name of the sequence (sequencde id). \\

    C is number of channels, N is number of images should be 10, H and W are height and width of images.
    """

    def __init__(
        self,
        split: str,
        manual_dataset_path=None,
    ) -> None:
        """
        Args:
            manual_dataset_path (str, optional): Path to dataset. Defaults to None.
        """
        super().__init__()

        if manual_dataset_path:
            self.path_to_dataset = os.path.join(manual_dataset_path, "Eval")
        else:
            self.path_to_dataset = os.path.join(os.getcwd(), "data", "Eval")

        # Load names of all sequences into dataframe with column name "id"
        self.df = pd.DataFrame(
            [int(x) for x in os.listdir(self.path_to_dataset) if x.isnumeric()],
            columns=["id"],
        )
        # Add directory paths to those ids
        self.df["dir"] = [
            os.path.join(self.path_to_dataset, str(x)) for x in self.df["id"]
        ]

    def __getitem__(self, index):
        dir = self.df.iloc[index]["dir"]

        ground_truth = os.path.join(dir, "gt.png")
        assert os.path.exists(ground_truth) == True, f"{ground_truth} does not exist"

        # Scale gt and remove alpha channel
        ground_truth = ToTensor()(Image.open(ground_truth).resize((256, 256)))[:3, :, :]

        # Read images, resize to 256 x 256, remove alpha channel and remove 0th image.
        images = torch.zeros((10, 3, 256, 256))

        for x in os.listdir(dir):
            if x.split(".")[0].isnumeric() and x.split(".")[0] != "0":
                file_i = int(x.split(".")[0])
                img = Image.open(os.path.join(dir, x))
                img = img.resize((256, 256))
                img = ToTensor()(img)

                img = img[:3, :, :]
                images[file_i - 1] = img

        images = torch.swapaxes(images, 0, 1)

        sl = torch.zeros_like(images)
        occ = torch.zeros_like(images)

        # get sequence id
        dir_name = dir.split("/")[-1]
        return (images, ground_truth, sl, occ, dir_name)

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
        split_version: str = "split-1_80_10_10",
        manual_dataset_path=None,
    ) -> None:
        """
        Args:
            split (str): train, val or test
            split_version (str, optional): Which split to use. Defaults to "split-1_80_10_10".
            manual_dataset_path (str, optional): Path to dataset. Defaults to None.
        """
        super().__init__(split, split_version, manual_dataset_path)

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
        split_version: str = "split-1_80_10_10",
        manual_dataset_path=None,
    ) -> None:
        """
        Args:
            split (str): train, val or test
            split_version (str, optional): Which split to use. Defaults to "split-1_80_10_10".
            manual_dataset_path (str, optional): Path to dataset. Defaults to None.
        """
        super().__init__(split, split_version, manual_dataset_path)

    def rearrange_targets(self, targets):
        rearranged_targets = torch.zeros_like(targets)
        rearranged_targets[:, 0, :, :] = targets[:, 9, :, :]
        rearranged_targets[:, 1, :, :] = targets[:, 2, :, :]
        rearranged_targets[:, 2, :, :] = targets[:, 7, :, :]
        rearranged_targets[:, 3, :, :] = targets[:, 8, :, :]
        rearranged_targets[:, 4, :, :] = targets[:, 3, :, :]
        rearranged_targets[:, 5, :, :] = targets[:, 4, :, :]
        rearranged_targets[:, 6, :, :] = targets[:, 6, :, :]
        rearranged_targets[:, 7, :, :] = targets[:, 5, :, :]
        rearranged_targets[:, 8, :, :] = targets[:, 0, :, :]
        rearranged_targets[:, 9, :, :] = targets[:, 1, :, :]

        return rearranged_targets

    def __getitem__(self, index):
        dir = self.df.iloc[index]["dir"]

        ground_truth = os.path.join(dir, "gt.png")
        assert os.path.exists(ground_truth) == True, f"{ground_truth} does not exist"
        ground_truth = ToTensor()(Image.open(ground_truth))

        images = os.listdir(dir)
        images.sort()
        # print(images)

        images = torch.stack(
            [
                ToTensor()(Image.open(os.path.join(dir, x)))
                for x in images
                if x.split(".")[0].isnumeric()
            ]
        )
        images = torch.swapaxes(images, 0, 1)

        sample = dir.split("/")[-1]
        path_to_occ = f"{self.path_to_dataset}/SIAR_OCC/{sample}.pt"
        occ = torch.load(path_to_occ)
        occ = self.rearrange_targets(occ)

        sl = torch.zeros_like(images)

        return (images, ground_truth, sl, occ)


class SIAR_OCC_Binary(SIAR):
    """
    SIAR Dataset with binary occlusion approximation targets.

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
        split_version: str = "split-1_80_10_10",
        manual_dataset_path=None,
    ) -> None:
        """
        Args:
            split (str): train, val or test
            split_version (str, optional): Which split to use. Defaults to "split-1_80_10_10".
            manual_dataset_path (str, optional): Path to dataset. Defaults to None.
        """
        super().__init__(split, split_version, manual_dataset_path)

    def rearrange_targets(self, targets):
        rearranged_targets = torch.zeros_like(targets)
        rearranged_targets[:, 0, :, :] = targets[:, 9, :, :]
        rearranged_targets[:, 1, :, :] = targets[:, 2, :, :]
        rearranged_targets[:, 2, :, :] = targets[:, 7, :, :]
        rearranged_targets[:, 3, :, :] = targets[:, 8, :, :]
        rearranged_targets[:, 4, :, :] = targets[:, 3, :, :]
        rearranged_targets[:, 5, :, :] = targets[:, 4, :, :]
        rearranged_targets[:, 6, :, :] = targets[:, 6, :, :]
        rearranged_targets[:, 7, :, :] = targets[:, 5, :, :]
        rearranged_targets[:, 8, :, :] = targets[:, 0, :, :]
        rearranged_targets[:, 9, :, :] = targets[:, 1, :, :]

        return rearranged_targets

    def __getitem__(self, index):
        dir = self.df.iloc[index]["dir"]

        ground_truth = os.path.join(dir, "gt.png")
        assert os.path.exists(ground_truth) == True, f"{ground_truth} does not exist"
        ground_truth = ToTensor()(Image.open(ground_truth))

        images = os.listdir(dir)
        images.sort()
        # print(images)

        images = torch.stack(
            [
                ToTensor()(Image.open(os.path.join(dir, x)))
                for x in images
                if x.split(".")[0].isnumeric()
            ]
        )
        images = torch.swapaxes(images, 0, 1)

        sample = dir.split("/")[-1]
        path_to_occ = f"{self.path_to_dataset}/SIAR_OCC/{sample}.pt"
        occ = torch.load(path_to_occ)
        occ = self.rearrange_targets(occ)

        # Turn into binary mask
        occ = occ.sum(dim=0)
        occ = torch.where(
            occ > 0.5,
            1.0,
            0.0,
        ).to(occ.device)

        sl = torch.zeros_like(images)

        return (images, ground_truth, sl, occ)


class SIAR_OCC_Binary_SUB(SIAR):
    """
    SIAR Dataset with binary occlusion approximation targets.

    Returns:
        images (torch.Tensor): [C, N, H, W] tensor of images.
        ground_truth (torch.Tensor): [C, H, W] tensor of ground truth image.
        sl (torch.Tensor): [C, N, H, W] tensor of shadow and light approximation. In this case we return zeros, as we don't need it in the OCC dataset.
        occ (torch.Tensor): [N, H, W] tensor of occlusion approximation. Loaded from pre-generated files.

    C is number of channels, N is number of images should be 10, H and W are height and width of images.
    """

    def __init__(
        self,
        split: str,
        split_version: str = "split-1_80_10_10",
        manual_dataset_path=None,
    ) -> None:
        """
        Args:
            split (str): train, val or test
            split_version (str, optional): Which split to use. Defaults to "split-1_80_10_10".
            manual_dataset_path (str, optional): Path to dataset. Defaults to None.
        """
        super().__init__(split, split_version, manual_dataset_path)

    def __getitem__(self, index):
        dir = self.df.iloc[index]["dir"]

        ground_truth = os.path.join(dir, "gt.png")
        assert os.path.exists(ground_truth) == True, f"{ground_truth} does not exist"
        ground_truth = ToTensor()(Image.open(ground_truth))

        images = images_to_tensor(dir)

        sample = dir.split("/")[-1]
        path_to_occ = f"{self.path_to_dataset}/SIAR_OCC_SUB/{sample}"
        occ = images_to_tensor(path_to_occ)

        # Turn into binary mask
        occ = occ.squeeze(0)
        # occ = torch.where(
        #     occ > 0.5,
        #     1.0,
        #     0.0,
        # ).to(occ.device)

        sl = torch.zeros_like(images)

        return (images, ground_truth, sl, occ)


class SIAR_BINARY_OCC_AND_SL(SIAR):
    """
    SIAR Dataset with binary occlusion targets and shadow and light targets.

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
        split_version: str = "split-1_80_10_10",
        manual_dataset_path=None,
    ) -> None:
        """
        Args:
            split (str): train, val or test
            split_version (str, optional): Which split to use. Defaults to "split-1_80_10_10".
            manual_dataset_path (str, optional): Path to dataset. Defaults to None.
        """
        super().__init__(split, split_version, manual_dataset_path)

    def rearrange_targets(self, targets):
        rearranged_targets = torch.zeros_like(targets)
        rearranged_targets[:, 0, :, :] = targets[:, 9, :, :]
        rearranged_targets[:, 1, :, :] = targets[:, 2, :, :]
        rearranged_targets[:, 2, :, :] = targets[:, 7, :, :]
        rearranged_targets[:, 3, :, :] = targets[:, 8, :, :]
        rearranged_targets[:, 4, :, :] = targets[:, 3, :, :]
        rearranged_targets[:, 5, :, :] = targets[:, 4, :, :]
        rearranged_targets[:, 6, :, :] = targets[:, 6, :, :]
        rearranged_targets[:, 7, :, :] = targets[:, 5, :, :]
        rearranged_targets[:, 8, :, :] = targets[:, 0, :, :]
        rearranged_targets[:, 9, :, :] = targets[:, 1, :, :]

        return rearranged_targets

    def __getitem__(self, index):
        dir = self.df.iloc[index]["dir"]

        ground_truth = os.path.join(dir, "gt.png")
        assert os.path.exists(ground_truth) == True, f"{ground_truth} does not exist"
        ground_truth = ToTensor()(Image.open(ground_truth))

        images = os.listdir(dir)
        images.sort()
        # print(images)

        images = torch.stack(
            [
                ToTensor()(Image.open(os.path.join(dir, x)))
                for x in images
                if x.split(".")[0].isnumeric()
            ]
        )
        images = torch.swapaxes(images, 0, 1)

        sample = dir.split("/")[-1]
        path_to_occ = f"{self.path_to_dataset}/SIAR_OCC/{sample}.pt"
        occ = torch.load(path_to_occ)
        occ = self.rearrange_targets(occ)

        # Turn into binary mask
        occ = occ.sum(dim=0)
        occ = torch.where(
            occ > 0.5,
            1.0,
            0.0,
        ).to(occ.device)

        sl = get_shadow_light_gt(ground_truth, images)

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
        split_version: str = "split-1_80_10_10",
        manual_dataset_path=None,
    ) -> None:
        """
        Args:
            split (str): train, val or test
            split_version (str, optional): Which split to use. Defaults to "split-1_80_10_10".
            manual_dataset_path (str, optional): Path to dataset. Defaults to None.
        """
        super().__init__(split, split_version, manual_dataset_path)

    def __getitem__(self, index):
        dir = self.df.iloc[index]["dir"]

        ground_truth = os.path.join(dir, "gt.png")
        assert os.path.exists(ground_truth) == True, f"{ground_truth} does not exist"
        ground_truth = ToTensor()(Image.open(ground_truth))
        images = torch.stack(
            [
                ToTensor()(Image.open(os.path.join(dir, x)))
                for x in sorted(os.listdir(dir))
                if x.split(".")[0].isnumeric()
            ]
        )
        images = torch.swapaxes(images, 0, 1)

        occ = get_occlusion_gt(ground_truth, images)
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

    def setup(self, stage: str) -> None:
        if stage == "test":
            self.siar_test = self.dataset(
                split="test",
                manual_dataset_path=self.manual_dataset_path,
            )
        elif stage == "train":
            self.siar_train = self.dataset(
                split="train",
                manual_dataset_path=self.manual_dataset_path,
            )
            self.siar_val = self.dataset(
                split="val",
                manual_dataset_path=self.manual_dataset_path,
            )

    def train_dataloader(self):
        return DataLoader(
            self.siar_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
        # return DataLoader(self.siar_train, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(
            self.siar_test, batch_size=self.batch_size, num_workers=self.num_workers
        )
        # return DataLoader(self.siar_test, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(
            self.siar_val, batch_size=self.batch_size, num_workers=self.num_workers
        )
        # return DataLoader(self.siar_val, batch_size=self.batch_size)
