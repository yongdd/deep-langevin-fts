import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl

class LitAtrousCascadeXception(pl.LightningModule): # Atrous-Cascade and Xception
    def __init__(self, dim, in_channels=3, mid_channels = 32, out_channels=1, kernel_size=3):
        super().__init__()
        padding = (kernel_size-1)//2
        self.dim = dim
            
        if dim == 3:
            self.conv1 = nn.Conv3d(in_channels,  mid_channels, kernel_size, padding_mode='circular', padding=padding)
            
            # Point-wise Seperable Convolutions
            self.conv2_point   = nn.Conv3d(mid_channels, mid_channels, 1)
            self.conv3_point   = nn.Conv3d(mid_channels, mid_channels, 1)
            
            self.conv4_1_point = nn.Conv3d(mid_channels, mid_channels, 1)
            self.conv4_2_point = nn.Conv3d(mid_channels, mid_channels, 1)
            
            self.conv5_point   = nn.Conv3d(mid_channels, mid_channels, 1)
            self.conv6_point   = nn.Conv3d(mid_channels, mid_channels, 1)
            self.conv7         = nn.Conv3d(mid_channels, out_channels, 1)
            
            # Depth-wise Seperable Convolutions
            self.conv2_depth   = nn.Conv3d(mid_channels, mid_channels, kernel_size, bias=False, groups=mid_channels, padding_mode='circular', padding=padding)
            self.conv3_depth   = nn.Conv3d(mid_channels, mid_channels, kernel_size, bias=False, groups=mid_channels, padding_mode='circular', padding=padding*2, dilation=2)
            
            self.conv4_1_depth = nn.Conv3d(mid_channels, mid_channels, kernel_size, bias=False, groups=mid_channels, padding_mode='circular', padding=padding*4, dilation=4)
            self.conv4_2_depth = nn.Conv3d(mid_channels, mid_channels, kernel_size, bias=False, groups=mid_channels, padding_mode='circular', padding=padding*8, dilation=8)
            
            self.conv5_depth   = nn.Conv3d(mid_channels, mid_channels, kernel_size, bias=False, groups=mid_channels, padding_mode='circular', padding=padding)
            self.conv6_depth   = nn.Conv3d(mid_channels, mid_channels, kernel_size, bias=False, groups=mid_channels, padding_mode='circular', padding=padding)
            
            # Batch Normalizations
            self.bm1 = nn.BatchNorm3d(mid_channels)
            self.bm2 = nn.BatchNorm3d(mid_channels)
            self.bm3 = nn.BatchNorm3d(mid_channels)
            
            self.bm4_1 = nn.BatchNorm3d(mid_channels)
            self.bm4_2 = nn.BatchNorm3d(mid_channels)
                        
            self.bm5 = nn.BatchNorm3d(mid_channels)
            self.bm6 = nn.BatchNorm3d(mid_channels)

    def forward(self, x):
        # Entry
        x1 = F.relu(self.bm1(self.conv1(x)))
        # Conv 2
        x = self.conv2_point(x1)
        x = F.relu(self.bm2(self.conv2_depth(x)))
        x1 = x1 + x
        # Conv 3
        x = self.conv3_point(x1)
        x = F.relu(self.bm3(self.conv3_depth(x)))
        x1 = x1 + x
        # Conv 4-1
        x = self.conv4_1_point(x1)
        x = F.relu(self.bm4_1(self.conv4_1_depth(x)))
        x1 = x1 + x
        # Conv 4-2
        x = self.conv4_2_point(x1)
        x = F.relu(self.bm4_2(self.conv4_2_depth(x)))
        x1 = x1 + x
        # Conv 5
        x = self.conv5_point(x1)
        x = F.relu(self.bm5(self.conv5_depth(x)))
        x1 = x1 + x
        # Conv 6
        x = self.conv6_point(x1)
        x = F.relu(self.bm6(self.conv6_depth(x)))
        x1 = x1 + x
        # Last
        x = self.conv7(x1)
        return x
