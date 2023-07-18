import sys
import os
import torch
import pandas as pd
from tqdm import tqdm
import cv2
import numpy as np

sys.path.append(f"{os.getcwd()}")
from src.data.siar_data import SIAR_OCC_GENERATION

def save_tensor(occlusions, sample):
    dir = os.path.join("data/SIAR_OCC", f"{sample}")
    os.makedirs(dir, exist_ok=True)
    for i in range(occlusions.shape[1]):
        if i == 1:
            name = 10
        elif i == 0:
            name = 1
        else:
            name = i
        occ = occlusions[:, i, :, :]
        occ = occ.moveaxis(0, 2)
        occ = occ * 255
        cv2.imwrite(os.path.join(dir, str(name) + ".png"), np.array(occ))

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
        if os.path.exists(os.path.join("data/SIAR_OCC", f"{sample}.tiff")):
            print(f"Skipping {sample}")
            continue

        _, _, _, occ = dataset_train[i]
        print("OCC: ", occ.shape)
        save_tensor(occ, sample)

    print("Val...")
    for i in tqdm(range(len(dataset_val))):
        sample = dataset_val.df.iloc[i]["dir"].split("/")[-1]
        if os.path.exists(os.path.join("data/SIAR_OCC", f"{sample}.tiff")):
            print(f"Skipping {sample}")
            continue

        _, _, _, occ = dataset_val[i]
        save_tensor(occ, sample)

    print("Test...")
    for i in tqdm(range(len(dataset_test))):
        sample = dataset_test.df.iloc[i]["dir"].split("/")[-1]
        if os.path.exists(os.path.join("data/SIAR_OCC", f"{sample}.tiff")):
            print(f"Skipping {sample}")
            continue

        _, _, _, occ = dataset_test[i]
        save_tensor(occ, sample)

if __name__ == "__main__":
    main()
