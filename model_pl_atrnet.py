import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl

class LitAtrNet(pl.LightningModule): # Atrous
    def __init__(self, dim, in_channels=3, mid_channels = 128, out_channels=1, kernel_size = 5):
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
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.conv6(x)
        return x
