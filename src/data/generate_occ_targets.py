import sys
import os
import torch
import pandas as pd
from tqdm import tqdm
import cv2
import numpy as np
import argparse

sys.path.append(f"{os.getcwd()}")
from src.data.siar_data import SIAR_OCC_GENERATION

def save_tensor(occlusions, sample, dir):
    dir_sample = os.path.join(dir, f"{sample}")
    os.makedirs(dir_sample, exist_ok=True)
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
        cv2.imwrite(os.path.join(dir_sample, str(name) + ".png"), np.array(occ))

def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument("--data", required = True, type = str)
    argparser.add_argument("--out", required = True, type = str)

    args = argparser.parse_args()
    return args

def main():
    args = get_args()
    manual_dataset_path = args.data
    """Generate the occ targets for the SIAR dataset. Store them in a new folder "data/SIAR_OCC" """
    dataset_train = SIAR_OCC_GENERATION(
        split="train",
        split_version="",
        sanity_check=False,
        manual_dataset_path=manual_dataset_path,
    )
    dataset_val = SIAR_OCC_GENERATION(
        split="val",
        split_version="",
        sanity_check=False,
        manual_dataset_path=manual_dataset_path,
    )
    dataset_test = SIAR_OCC_GENERATION(
        split="test",
        split_version="",
        sanity_check=False,
        manual_dataset_path=manual_dataset_path,
    )
    #dir = os.path.join(args.out, f"{sample}")
    os.makedirs(dir, exist_ok=True)

    print("Generating occ targets...")
    print("Train...")
    for i in tqdm(range(len(dataset_train))):
        sample = dataset_train.df.iloc[i]["dir"].split("/")[-1]
        if os.path.exists(os.path.join("data/SIAR_OCC", f"{sample}.tiff")):
            print(f"Skipping {sample}")
            continue

        _, _, _, occ = dataset_train[i]
        print("OCC: ", occ.shape)
        save_tensor(occ, sample, dir)

    print("Val...")
    for i in tqdm(range(len(dataset_val))):
        sample = dataset_val.df.iloc[i]["dir"].split("/")[-1]
        if os.path.exists(os.path.join("data/SIAR_OCC", f"{sample}.tiff")):
            print(f"Skipping {sample}")
            continue

        _, _, _, occ = dataset_val[i]
        save_tensor(occ, sample, dir)

    print("Test...")
    for i in tqdm(range(len(dataset_test))):
        sample = dataset_test.df.iloc[i]["dir"].split("/")[-1]
        if os.path.exists(os.path.join("data/SIAR_OCC", f"{sample}.tiff")):
            print(f"Skipping {sample}")
            continue

        _, _, _, occ = dataset_test[i]
        save_tensor(occ, sample, dir)

if __name__ == "__main__":
    main()
