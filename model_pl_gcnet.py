import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl

class LitGCNet(pl.LightningModule): # Global Convolutional Network
    def __init__(self, dim, in_channels=3, mid_channels = 132, out_channels=1, kernel_size = 5):
        super().__init__()
        padding = (kernel_size-1)//2
        g_kernel_size = 11
        g_padding = (g_kernel_size-1)//2
        self.dim = dim
        
        if dim == 1:
            self.conv1 = nn.Conv1d(in_channels,  mid_channels, kernel_size, padding_mode='circular', padding=padding)
            self.conv2 = nn.Conv1d(mid_channels, mid_channels, kernel_size, padding_mode='circular', padding=padding)
            
            self.conv3 = nn.Conv1d(mid_channels, mid_channels, kernel_size=g_kernel_size, padding_mode='circular', padding=g_padding*2, dilation=2)
            
            self.conv4 = nn.Conv1d(mid_channels, mid_channels, kernel_size, padding_mode='circular', padding=padding)          
            self.conv5 = nn.Conv1d(mid_channels, out_channels, 1)
            
        elif dim == 2:
            self.conv1 = nn.Conv2d(in_channels,  mid_channels, kernel_size, padding_mode='circular', padding=padding)
            self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size, padding_mode='circular', padding=padding)
            
            self.conv3_11 = nn.Conv2d(mid_channels, mid_channels, kernel_size=(1,g_kernel_size), padding_mode='circular', padding=(0,g_padding*4), dilation=4)
            self.conv3_12 = nn.Conv2d(mid_channels, mid_channels, kernel_size=(g_kernel_size,1), padding_mode='circular', padding=(g_padding*4,0), dilation=4)
            self.conv3_21 = nn.Conv2d(mid_channels, mid_channels, kernel_size=(g_kernel_size,1), padding_mode='circular', padding=(g_padding*4,0), dilation=4)
            self.conv3_22 = nn.Conv2d(mid_channels, mid_channels, kernel_size=(1,g_kernel_size), padding_mode='circular', padding=(0,g_padding*4), dilation=4)
             
            self.conv4 = nn.Conv2d(mid_channels, mid_channels, kernel_size, padding_mode='circular', padding=padding)          
            self.conv5 = nn.Conv2d(mid_channels, out_channels, 1)
            
        elif dim == 3:
            self.conv1 = nn.Conv3d(in_channels,  mid_channels, kernel_size, padding_mode='circular', padding=padding)
            self.conv2 = nn.Conv3d(mid_channels, mid_channels, kernel_size, padding_mode='circular', padding=padding)
            

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        if self.dim == 1:
            x = F.relu(self.conv3(x))
        elif self.dim == 2:
            x1 = F.relu(self.conv3_11(x))
            x1 = F.relu(self.conv3_12(x1))
            x2 = F.relu(self.conv3_21(x))
            x2 = F.relu(self.conv3_22(x2))
            x = F.relu(x1 + x2)
            
        x = F.relu(self.conv4(x))
        x = self.conv5(x)
        return x
