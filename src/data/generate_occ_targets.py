import sys
import os
import torch
import pandas as pd
from tqdm import tqdm

sys.path.append(f"{os.getcwd()}")
from src.data.siar_data import SIAR_OCC


def main():
    """Generate the occ targets for the SIAR dataset. Store them in a new folder "data/SIAR_OCC" """
    dataset_train = SIAR_OCC(
        split="train",
        split_version="split-1_80_10_10",
        sanity_check=False,
        manual_dataset_path=None,
    )
    dataset_val = SIAR_OCC(
        split="val",
        split_version="split-1_80_10_10",
        sanity_check=False,
        manual_dataset_path=None,
    )
    dataset_test = SIAR_OCC(
        split="test",
        split_version="split-1_80_10_10",
        sanity_check=False,
        manual_dataset_path=None,
    )

    if not os.path.exists("data/SIAR_OCC"):
        os.makedirs("data/SIAR_OCC")

    print("Generating occ targets...")
    print("Train...")
    for i in tqdm(range(len(dataset_train))):
        _, _, _, occ = dataset_train[i]
        sample = dataset_train.df.iloc[i]["dir"].split("/")[-1]
        torch.save(occ, os.path.join("data/SIAR_OCC", f"{sample}.pt"))

    print("Val...")
    for i in tqdm(range(len(dataset_val))):
        _, _, _, occ = dataset_val[i]
        sample = dataset_val.df.iloc[i]["dir"].split("/")[-1]
        torch.save(occ, os.path.join("data/SIAR_OCC", f"{sample}.pt"))

    print("Test...")
    for i in tqdm(range(len(dataset_test))):
        _, _, _, occ = dataset_test[i]
        sample = dataset_test.df.iloc[i]["dir"].split("/")[-1]
        torch.save(occ, os.path.join("data/SIAR_OCC", f"{sample}.pt"))


if __name__ == "__main__":
    main()
