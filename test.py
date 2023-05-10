import torch
from data.dataloader import DecompositionDataModule
from decomposer import Decomposer
import lightning.pytorch as pl


dataModule = DecompositionDataModule(datapath="data/SIAR", batch_size=2)
dataModule.setup(stage='fit')
model =  Decomposer()
trainer = pl.Trainer(max_epochs=1)
trainer.fit(model, train_dataloaders=dataModule.train_dataloader())