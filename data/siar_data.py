import os
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

    def __init__(self, data_dir: str, split: str) -> None:
        super().__init__()

        self.df = pd.read_csv(os.path.join(data_dir, split + '.csv'), index_col=None)
        self.df['dir'] = [os.path.join(data_dir, x) for x in self.df['id']]

    def __getitem__(self, index):
        dir = self.df.iloc[index]['dir']

        ground_truth = os.path.join(dir, 'gt.png')
        assert os.path.exists(ground_truth) == True
        ground_truth = ToTensor()(Image.open(ground_truth))
        
        images = torch.stack([ToTensor()(Image.open(os.path.join(dir, x))) for x in os.listdir(dir) if x.split('.')[0].isnumeric()])
        images = torch.swapaxes(images, 0, 1)

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
            self.siar_train = SIAR(self.data_dir, 'train')
        if stage == 'test':
            self.siar_test = SIAR(self.data_dir, 'test')
        if stage == 'validate':
            self.siar_val = SIAR(self.data_dir, 'val')
        # etc...

    def train_dataloader(self):
        return DataLoader(self.siar_train, batch_size=self.batch_size)
    
    def test_dataloader(self):
        return DataLoader(self.siar_train, batch_size=self.batch_size)
    
    def val_dataloader(self):
        return DataLoader(self.siar_train, batch_size=self.batch_size)