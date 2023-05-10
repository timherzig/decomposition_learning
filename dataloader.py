import torch
from torch.utils.data import random_split, DataLoader
from torch.utils.data import Dataset
import lightning.pytorch as pl
import os
import re
import natsort
import numpy as np
from skimage import io
import plotly.express as px


class ImageDataset(Dataset):
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.data = os.listdir(data_dir)

    def __len__(self):
        return(len(self.data))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()   
      
        # get the list of images in each image folder in the data directory
        items = os.listdir(os.path.join(self.data_dir, self.data[idx]))

        # get image sequence of shape T x H x W x C
        image_list = []
        pattern = r'\d+'
        for item in items:
            if re.search(pattern, item):
                image = io.imread(os.path.join(self.data_dir, self.data[idx], item))
                image_list.append(image)      
        image_sequence = np.stack(image_list, axis = 0)  

        # get ground truth label
        for item in items: 
            if item.startswith('gt'):
                label = io.imread(os.path.join(self.data_dir, self.data[idx], item))

        sample = image_sequence, label

        return sample


# https://lightning.ai/docs/pytorch/stable/data/datamodule.html

class ImageDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.data = ImageDataset(data_dir)

    #def setup(self, stage: str):       
        self.image_train, self.image_val, self.image_test = random_split(self.data, [0.6, 0.2, 0.2])

    def train_dataloader(self):
        return DataLoader(self.image_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.image_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.image_test, batch_size=self.batch_size)

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        ...
        

if __name__ == "__main__":
    data_module = ImageDataModule('/content/data', 4)
    trainloader = data_module.train_dataloader()
