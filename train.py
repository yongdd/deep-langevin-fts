import os
import time
import random
import pathlib
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from pytorch_lightning.plugins import DDPPlugin
from dataset import *

from model.unet import *
from model.aspp import *
from model.atrpar import *
from model.atrcas import *
from model.atrcasx import *

class TrainerAndModel(LitAtrPar): # LitUNet, LitASPP, LitAtrPar, LitAtrCas, LitAtrCasX
    def __init__(self, dim, mid_channels):
        super().__init__(dim=dim, mid_channels=mid_channels)
        self.loss = torch.nn.MSELoss()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[100], gamma=0.2,
            verbose=False)
        return [optimizer], [scheduler]
    
    def on_train_start(self):
        total_params = sum(p.numel() for p in self.parameters())
        self.log('total_params', float(total_params))
        print("total_params", total_params)
    
    def on_epoch_start(self):
        self.log('learning_rate', self.optimizers().param_groups[0]['lr'])
        #print('\n')

    def on_epoch_end(self):
        path = "saved_model_weights"
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, 'epoch_%d.pth' % (self.current_epoch)))
      
    def training_step(self, train_batch, batch_idx):
        x = train_batch['input']
        y = train_batch['target']
        x = self(x)   
        loss = self.loss(y, x)
        self.log('train_loss', loss)
        return loss

if __name__=="__main__":

    os.environ["PL_TORCH_DISTRIBUTED_BACKEND"]="gloo" #nccl or gloo
    #os.environ["CUDA_VISIBLE_DEVICES"]= "5,6"
    torch.set_num_threads(1)

    data_dir = "data_training"
    model = TrainerAndModel(dim=3, mid_channels=32)
    
    # training data    
    train_dataset = FtsDataset(data_dir)
    train_loader = DataLoader(train_dataset, batch_size=8, num_workers=4)
    print(len(train_dataset))
    
    # training
    trainer = pl.Trainer(
            gpus=1, num_nodes=1, max_epochs=100, precision=16,
            strategy=DDPPlugin(find_unused_parameters=False),
            benchmark=True, log_every_n_steps=5)
    trainer.fit(model, train_loader, None)