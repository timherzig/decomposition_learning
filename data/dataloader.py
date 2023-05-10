
from torch.utils.data import random_split,  DataLoader
import pytorch_lightning as pl
from data.dataset import DecompositionDataset
from torchvision import transforms

class DecompositionDataModule(pl.LightningDataModule):
    """
    DataModule used decomposition project
    """

    def __init__(self, datapath, batch_size=32):
        super(DecompositionDataModule).__init__()
        self.datapath = datapath
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.batch_size = batch_size

    def prepare_data(self):
        pass

    def setup(self, stage: str):        
        if stage == "fit":
            train_set =  DecompositionDataset(datapath=self.datapath, train=True, transform=self.transform)
            train_set_size = int(len(train_set) * 0.9)
            valid_set_size = len(train_set) - train_set_size
            self.train, self.validate = random_split(train_set, [train_set_size, valid_set_size])

        if stage == "test":
            test_set =  DecompositionDataset(datapath=self.datapath, train=False, transform=self.transform)
            self.test = test_set

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=1)

    def val_dataloader(self):
        return DataLoader(self.validate, batch_size=self.batch_size, num_workers=1)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=1)

