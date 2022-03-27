import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl

class LitAtrousParallel(pl.LightningModule): # Atrous-Parallel
    def __init__(self, dim, in_channels=3, mid_channels=32, out_channels=1, kernel_size=3):
        super().__init__()
        padding = (kernel_size-1)//2
            
        if dim == 3:
            self.conv1 = nn.Conv3d(in_channels,  mid_channels, kernel_size, bias=False, padding=padding, padding_mode='circular')
            self.conv2 = nn.Conv3d(mid_channels, mid_channels, kernel_size, bias=False, padding=padding, padding_mode='circular')
            self.conv3 = nn.Conv3d(mid_channels, mid_channels, kernel_size, bias=False, padding=padding*2, padding_mode='circular', dilation=2)
            
            self.conv_aspp4_1 = nn.Conv3d(mid_channels, mid_channels//2, 1, bias=False)
            self.conv_aspp4_2 = nn.Conv3d(mid_channels, mid_channels//2, kernel_size, bias=False, padding=padding*2, padding_mode='circular', dilation=2)
            self.conv_aspp4_3 = nn.Conv3d(mid_channels, mid_channels//2, kernel_size, bias=False, padding=padding*4, padding_mode='circular', dilation=4)
            self.conv_aspp4_4 = nn.Conv3d(mid_channels, mid_channels//2, kernel_size, bias=False, padding=padding*8, padding_mode='circular', dilation=8)

            self.conv5 = nn.Conv3d(mid_channels*2, mid_channels, kernel_size, bias=False, padding=padding, padding_mode='circular')
            self.conv6 = nn.Conv3d(mid_channels,   mid_channels, kernel_size, bias=False, padding=padding, padding_mode='circular')
            self.conv7 = nn.Conv3d(mid_channels,   out_channels, 1)
            
            self.bm1   = nn.BatchNorm3d(mid_channels)
            self.bm2   = nn.BatchNorm3d(mid_channels)
            self.bm3   = nn.BatchNorm3d(mid_channels)
            self.bm4_1 = nn.BatchNorm3d(mid_channels//2)
            self.bm4_2 = nn.BatchNorm3d(mid_channels//2)
            self.bm4_3 = nn.BatchNorm3d(mid_channels//2)
            self.bm4_4 = nn.BatchNorm3d(mid_channels//2)
            self.bm5   = nn.BatchNorm3d(mid_channels)
            self.bm6   = nn.BatchNorm3d(mid_channels)

    def forward(self, x):
        x = F.relu(self.bm1(self.conv1(x)))
        x = F.relu(self.bm2(self.conv2(x)))
        x = F.relu(self.bm3(self.conv3(x)))
        
        x1 = F.relu(self.bm4_1(self.conv_aspp4_1(x)))
        x2 = F.relu(self.bm4_2(self.conv_aspp4_2(x)))
        x3 = F.relu(self.bm4_3(self.conv_aspp4_3(x)))
        x4 = F.relu(self.bm4_4(self.conv_aspp4_4(x)))

        x = torch.cat([x1, x2, x3, x4],dim=1)
        x = F.relu(self.bm5(self.conv5(x)))
        x = F.relu(self.bm6(self.conv6(x)))
        x  = self.conv7(x)
        return x
