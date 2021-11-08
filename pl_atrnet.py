import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl

class LitAtrNet(pl.LightningModule):
    def __init__(self, dim, in_channels=3, mid_channels = 64, out_channels=1, kernel_size = 3):
        super().__init__()
        padding = (kernel_size-1)//2
        
        if dim == 1:
            self.conv1 = nn.Conv1d(in_channels,  mid_channels, kernel_size, padding=padding, padding_mode='circular')
            self.conv2 = nn.Conv1d(mid_channels, mid_channels, kernel_size, padding=padding, padding_mode='circular')
            self.conv3 = nn.Conv1d(mid_channels, mid_channels, kernel_size, padding=padding*2, padding_mode='circular', dilation=2)
            self.conv4 = nn.Conv1d(mid_channels, mid_channels, kernel_size, padding=padding*4, padding_mode='circular', dilation=4)
            self.conv5 = nn.Conv1d(mid_channels, mid_channels, kernel_size, padding=padding*8, padding_mode='circular', dilation=8)
            self.conv6 = nn.Conv1d(mid_channels, out_channels, 1)
            
        elif dim == 2:
            self.conv1 = nn.Conv2d(in_channels,  mid_channels, kernel_size, padding=padding, padding_mode='circular')
            self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size, padding=padding, padding_mode='circular')
            self.conv3 = nn.Conv2d(mid_channels, mid_channels, kernel_size, padding=padding*2, padding_mode='circular', dilation=2)
            self.conv4 = nn.Conv2d(mid_channels, mid_channels, kernel_size, padding=padding*4, padding_mode='circular', dilation=4)
            self.conv5 = nn.Conv2d(mid_channels, mid_channels, kernel_size, padding=padding*8, padding_mode='circular', dilation=8)
            self.conv6 = nn.Conv2d(mid_channels, out_channels, 1)
            
        elif dim == 3:
            self.conv1 = nn.Conv3d(in_channels,  mid_channels, kernel_size, padding=padding, padding_mode='circular')
            self.conv2 = nn.Conv3d(mid_channels, mid_channels, kernel_size, padding=padding, padding_mode='circular')
            self.conv3 = nn.Conv3d(mid_channels, mid_channels, kernel_size, padding=padding*2, padding_mode='circular', dilation=2)
            self.conv4 = nn.Conv3d(mid_channels, mid_channels, kernel_size, padding=padding*4, padding_mode='circular', dilation=4)
            self.conv5 = nn.Conv3d(mid_channels, mid_channels, kernel_size, padding=padding*8, padding_mode='circular', dilation=8)
            self.conv7 = nn.Conv3d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        x = nn.functional.relu(self.conv4(x))
        x = nn.functional.relu(self.conv5(x))
        x = self.conv6(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[10,20,30,100], gamma=0.5,
            verbose=False)
        return [optimizer], [scheduler]
    
    def on_epoch_start(self):
        self.log('learning_rate', self.optimizers().param_groups[0]['lr'])
        print('\n')
        
    def on_epoch_end(self):
        torch.save(self.state_dict(), f'saved_model_{self.current_epoch}.pth')
        
    def training_step(self, train_batch, batch_idx):
        x = train_batch['data']
        y = train_batch['target']
        x = self(x)   
        loss = F.mse_loss(y, x)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x = val_batch['data']
        y = val_batch['target']
        x = self(x)  
        loss = F.mse_loss(y, x)
        self.log('val_loss', loss)
