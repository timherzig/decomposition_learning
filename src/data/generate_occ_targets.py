import sys
import os
import torch
import pandas as pd
from tqdm import tqdm

sys.path.append(f"{os.getcwd()}")
from src.data.siar_data import SIAR_OCC_GENERATION


def main():
    """Generate the occ targets for the SIAR dataset. Store them in a new folder "data/SIAR_OCC" """
    dataset_train = SIAR_OCC_GENERATION(
        split="train",
        split_version="split-1_80_10_10",
        sanity_check=False,
        manual_dataset_path=None,
    )
    dataset_val = SIAR_OCC_GENERATION(
        split="val",
        split_version="split-1_80_10_10",
        sanity_check=False,
        manual_dataset_path=None,
    )
    dataset_test = SIAR_OCC_GENERATION(
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
        sample = dataset_train.df.iloc[i]["dir"].split("/")[-1]
        if os.path.exists(os.path.join("data/SIAR_OCC", f"{sample}.pt")):
            print(f"Skipping {sample}")
            continue

        _, _, _, occ = dataset_train[i]
        torch.save(occ, os.path.join("data/SIAR_OCC", f"{sample}.pt"))

    print("Val...")
    for i in tqdm(range(len(dataset_val))):
        sample = dataset_val.df.iloc[i]["dir"].split("/")[-1]
        if os.path.exists(os.path.join("data/SIAR_OCC", f"{sample}.pt")):
            print(f"Skipping {sample}")
            continue

        _, _, _, occ = dataset_val[i]
        torch.save(occ, os.path.join("data/SIAR_OCC", f"{sample}.pt"))

    print("Test...")
    for i in tqdm(range(len(dataset_test))):
        sample = dataset_test.df.iloc[i]["dir"].split("/")[-1]
        if os.path.exists(os.path.join("data/SIAR_OCC", f"{sample}.pt")):
            print(f"Skipping {sample}")
            continue

        _, _, _, occ = dataset_test[i]
        torch.save(occ, os.path.join("data/SIAR_OCC", f"{sample}.pt"))


if __name__ == "__main__":
    main()
