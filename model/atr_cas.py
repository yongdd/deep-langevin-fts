import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl

class LitAtrousCascade(pl.LightningModule): # Atrous-Cascade
    def __init__(self, dim, in_channels=3, mid_channels=32, out_channels=1, kernel_size = 3):
        super().__init__()
        padding = (kernel_size-1)//2
        self.dim = dim
        
        if dim == 3:
            self.conv1   = nn.Conv3d(in_channels,  mid_channels, kernel_size, bias=False, padding=padding, padding_mode='circular')
            self.conv2   = nn.Conv3d(mid_channels, mid_channels, kernel_size, bias=False, padding=padding, padding_mode='circular')
            self.conv3   = nn.Conv3d(mid_channels, mid_channels, kernel_size, bias=False, padding=padding*2, padding_mode='circular', dilation=2)
            self.conv4_1 = nn.Conv3d(mid_channels, mid_channels, kernel_size, bias=False, padding=padding*4, padding_mode='circular', dilation=4)
            self.conv4_2 = nn.Conv3d(mid_channels, mid_channels, kernel_size, bias=False, padding=padding*8, padding_mode='circular', dilation=8)
            self.conv5   = nn.Conv3d(mid_channels, mid_channels, kernel_size, bias=False, padding=padding, padding_mode='circular')
            self.conv6   = nn.Conv3d(mid_channels, mid_channels, kernel_size, bias=False, padding=padding, padding_mode='circular')
            self.conv7   = nn.Conv3d(mid_channels, out_channels, 1)

            self.bm1   = nn.BatchNorm3d(mid_channels)
            self.bm2   = nn.BatchNorm3d(mid_channels)
            self.bm3   = nn.BatchNorm3d(mid_channels)
            self.bm4_1 = nn.BatchNorm3d(mid_channels)
            self.bm4_2 = nn.BatchNorm3d(mid_channels)
            self.bm5   = nn.BatchNorm3d(mid_channels)
            self.bm6   = nn.BatchNorm3d(mid_channels)

    def forward(self, x):
        x = F.relu(self.bm1(self.conv1(x)))
        x = F.relu(self.bm2(self.conv2(x)))
        x = F.relu(self.bm3(self.conv3(x)))
        x = F.relu(self.bm4_1(self.conv4_1(x)))
        x = F.relu(self.bm4_2(self.conv4_2(x)))
        x = F.relu(self.bm5(self.conv5(x)))
        x = F.relu(self.bm6(self.conv6(x)))
        x = self.conv7(x)
        return x
