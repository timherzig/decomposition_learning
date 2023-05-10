import torch

from torch import nn
# from lightning.pytorch import LightningModule
from src.swin.swin_transformer import SwinTransformer3D

class Model(SwinTransformer3D):
    '''
    '''

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.swin_encoder = SwinTransformer3D()

    def forward(self, x):
        return

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        return opt
    
    def training_step(self, batch, batch_idx):
        # This is of course not final :)
        x, y = batch

        loss = 0
        self.log('train_loss', loss)
        return loss
    
    # def backward(self, trainer, loss, optimizer, optimizer_idx):
    #     loss.backward()