import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

class DecompositionDataset(Dataset):
    def __init__(self, datapath, train, transform = None):
        self.sequences = []
        # Iterate over dataset
        files = os.listdir(datapath)

        # Split dataset into train and test datasets ->13617:2000
        if(train):
            files = files[:-2000]
        else:
            files = files[-2000:]
        for file_name in os.listdir(datapath):
            file = os.path.join(datapath, file_name)

            self.sequences.append(file)
        self.transform = transform  
    def __len__(self):
        return len(self.sequences)    
    
    def __getitem__(self, idx):
        # Resize image
        sequence = np.zeros((10, 3, 256, 256))
        sequence_path = self.sequences[idx]
        for idx, img_name in enumerate(os.listdir(sequence_path)):
            img_path = os.path.join(sequence_path, img_name)
            name = img_name.split(".")[0]
            img = Image.open(img_path)
            img = np.array(img).astype(np.uint8)
            if(self.transform):
                img = self.transform(img)
            if(name == "gt"):
                ground_truth = img
            else:
                sequence[idx] = img
        sequence = np.moveaxis(sequence, 0, 1)
        return sequence, ground_truth