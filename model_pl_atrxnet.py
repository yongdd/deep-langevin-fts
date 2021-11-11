import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl

class LitAtrXNet(pl.LightningModule): # Atrous and Xception
    def __init__(self, dim, in_channels=3, mid_channels = 128, out_channels=1, kernel_size = 5):
        super().__init__()
        padding = (kernel_size-1)//2
        
        if dim == 1:
            self.conv1 = nn.Conv1d(in_channels,  mid_channels, kernel_size, padding_mode='circular', padding=padding)
            self.conv2 = nn.Conv1d(mid_channels, mid_channels, kernel_size, padding_mode='circular', padding=padding)
            
            self.conv3_point = nn.Conv1d(mid_channels, mid_channels, 1)
            self.conv4_point = nn.Conv1d(mid_channels, mid_channels, 1)
            self.conv5_point = nn.Conv1d(mid_channels, mid_channels, 1)
            
            self.conv3_depth = nn.Conv1d(mid_channels, mid_channels, kernel_size, groups=mid_channels, padding_mode='circular', padding=padding*2, dilation=2)
            self.conv4_depth = nn.Conv1d(mid_channels, mid_channels, kernel_size, groups=mid_channels, padding_mode='circular', padding=padding*4, dilation=4)
            self.conv5_depth = nn.Conv1d(mid_channels, mid_channels, kernel_size, groups=mid_channels, padding_mode='circular', padding=padding*8, dilation=8)
            
            self.conv6 = nn.Conv1d(mid_channels, mid_channels, kernel_size, padding_mode='circular', padding=padding)          
            self.conv7 = nn.Conv1d(mid_channels, out_channels, 1)
            
        elif dim == 2:
            self.conv1 = nn.Conv2d(in_channels,  mid_channels, kernel_size, padding_mode='circular', padding=padding)
            self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size, padding_mode='circular', padding=padding)
            
            self.conv3_point = nn.Conv2d(mid_channels, mid_channels, 1)
            self.conv4_point = nn.Conv2d(mid_channels, mid_channels, 1)
            self.conv5_point = nn.Conv2d(mid_channels, mid_channels, 1)
            
            self.conv3_depth = nn.Conv2d(mid_channels, mid_channels, kernel_size, groups=mid_channels, padding_mode='circular', padding=padding*2, dilation=2)
            self.conv4_depth = nn.Conv2d(mid_channels, mid_channels, kernel_size, groups=mid_channels, padding_mode='circular', padding=padding*4, dilation=4)
            self.conv5_depth = nn.Conv2d(mid_channels, mid_channels, kernel_size, groups=mid_channels, padding_mode='circular', padding=padding*8, dilation=8)
            
            self.conv6 = nn.Conv2d(mid_channels, mid_channels, kernel_size, padding_mode='circular', padding=padding)            
            self.conv7 = nn.Conv2d(mid_channels, out_channels, 1)
            
        elif dim == 3:
            self.conv1 = nn.Conv3d(in_channels,  mid_channels, kernel_size, padding_mode='circular', padding=padding)
            self.conv2 = nn.Conv3d(mid_channels, mid_channels, kernel_size, padding_mode='circular', padding=padding)
            
            self.conv3_point = nn.Conv3d(mid_channels, mid_channels, 1)
            self.conv4_point = nn.Conv3d(mid_channels, mid_channels, 1)
            self.conv5_point = nn.Conv3d(mid_channels, mid_channels, 1)
            
            self.conv3_depth = nn.Conv3d(mid_channels, mid_channels, kernel_size, groups=mid_channels, padding_mode='circular', padding=padding*2, dilation=2)
            self.conv4_depth = nn.Conv3d(mid_channels, mid_channels, kernel_size, groups=mid_channels, padding_mode='circular', padding=padding*4, dilation=4)
            self.conv5_depth = nn.Conv3d(mid_channels, mid_channels, kernel_size, groups=mid_channels, padding_mode='circular', padding=padding*8, dilation=8)
            
            self.conv6 = nn.Conv3d(mid_channels, mid_channels, kernel_size, padding_mode='circular', padding=padding)          
            self.conv7 = nn.Conv3d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        x = F.relu(self.conv3_point(x))
        x = F.relu(self.conv3_depth(x))
        x = F.relu(self.conv4_point(x))
        x = F.relu(self.conv4_depth(x))
        x = F.relu(self.conv5_point(x))
        x = F.relu(self.conv5_depth(x))
        
        x = F.relu(self.conv6(x))
        x = self.conv7(x)
        return x
