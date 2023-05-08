import os
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
import torch

import pandas as pd

from PIL import Image
from lightning.pytorch import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor

class SIAR(Dataset):
    '''
    PyTorch Dataset for the provided dataset
    '''

    def __init__(self, data_dir: str) -> None:
        super().__init__()

        self.df = pd.DataFrame(os.listdir(data_dir), columns=['id'])
        self.df['dir'] = [os.path.join(data_dir, x) for x in self.df['id']]

    def __getitem__(self, index):
        dir = self.df.iloc[[index]]['dir']

        ground_truth = os.path.join(dir, 'gt.png')
        assert os.path.exists(ground_truth) == True
        ground_truth = ToTensor()(Image.open(ground_truth))
        
        images = torch.Stack([ToTensor()(Image.open(x)) for x in os.listdir(dir) if x.split('.')[0].isnumeric()])

        return images, ground_truth

    def __len__(self):
        return len(self.df)
    

class SIARDataModule(LightningDataModule):
    '''
    Data Module for the 
    '''

    def __init__(self, data_dir: str, batch_size: int) -> None:
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size

    def prepare_data(self) -> None:
        return
    
    def setup(self, stage: str) -> None:
        if stage == 'train':
            self.siar_train = SIAR(self.data_dir)
        elif stage == 'test':
            self.siar_test = SIAR(self.data_dir)
        elif stage == 'validate':
            self.siar_val = SIAR(self.data_dir)
        # etc...

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.siar_train, batch_size=self.batch_size)
    
    def test_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.siar_train, batch_size=self.batch_size)
    
    def val_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.siar_train, batch_size=self.batch_size)