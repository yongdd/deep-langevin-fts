import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl

class LitDeepLab(pl.LightningModule):
    def __init__(self, dim, in_channels=3, mid_channels = 64, out_channels=1, kernel_size = 3):
        super().__init__()
        padding = (kernel_size-1)//2
        half_channels = mid_channels//2
        
        if dim == 1:
            self.conv1 = nn.Conv1d(in_channels,  mid_channels, kernel_size=5, padding=2, padding_mode='circular')
            self.conv2 = nn.Conv1d(mid_channels, mid_channels, kernel_size=5, padding=2, padding_mode='circular')
            
            self.conv3_point = nn.Conv1d(mid_channels, mid_channels, 1)
            self.conv4_point = nn.Conv1d(mid_channels, mid_channels, 1)
            self.conv5_point = nn.Conv1d(mid_channels, mid_channels, 1)
            self.conv6_point = nn.Conv1d(mid_channels, mid_channels, 1)
            
            self.conv3_depth = nn.Conv1d(mid_channels, mid_channels, kernel_size=5, groups=mid_channels, padding_mode='circular', padding=2*2, dilation=2)
            self.conv4_depth = nn.Conv1d(mid_channels, mid_channels, kernel_size=5, groups=mid_channels, padding_mode='circular', padding=2*2, dilation=2)
            self.conv5_depth = nn.Conv1d(mid_channels, mid_channels, kernel_size=5, groups=mid_channels, padding_mode='circular', padding=2*2, dilation=2)
            self.conv6_depth = nn.Conv1d(mid_channels, mid_channels, kernel_size=5, groups=mid_channels, padding_mode='circular', padding=2*2, dilation=2)
            
            self.conv_aspp1 = nn.Conv1d(mid_channels, half_channels, kernel_size, padding=padding*2, padding_mode='circular', dilation=2)
            self.conv_aspp2 = nn.Conv1d(mid_channels, half_channels, kernel_size, padding=padding*4, padding_mode='circular', dilation=4)
            self.conv_aspp3 = nn.Conv1d(mid_channels, half_channels, kernel_size, padding=padding*8, padding_mode='circular', dilation=8)
            self.conv_aspp4 = nn.Conv1d(mid_channels, half_channels, kernel_size, padding=padding*16, padding_mode='circular', dilation=16)
            
            self.conv7 = nn.Conv1d(half_channels*4, mid_channels, kernel_size=5, padding=2, padding_mode='circular')
            self.conv8 = nn.Conv1d(mid_channels, mid_channels, kernel_size=5, padding=2, padding_mode='circular')
            self.conv9 = nn.Conv1d(mid_channels, out_channels, 1)
            
        elif dim == 2:
            self.conv1 = nn.Conv2d(in_channels,  mid_channels, kernel_size=5, padding=2, padding_mode='circular')
            self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=5, padding=2, padding_mode='circular')
            
            self.conv3_point = nn.Conv2d(mid_channels, mid_channels, 1)
            self.conv4_point = nn.Conv2d(mid_channels, mid_channels, 1)
            self.conv5_point = nn.Conv2d(mid_channels, mid_channels, 1)
            self.conv6_point = nn.Conv2d(mid_channels, mid_channels, 1)
            
            self.conv3_depth = nn.Conv2d(mid_channels, mid_channels, kernel_size=5, groups=mid_channels, padding_mode='circular', padding=2*2, dilation=2)
            self.conv4_depth = nn.Conv2d(mid_channels, mid_channels, kernel_size=5, groups=mid_channels, padding_mode='circular', padding=2*2, dilation=2)
            self.conv5_depth = nn.Conv2d(mid_channels, mid_channels, kernel_size=5, groups=mid_channels, padding_mode='circular', padding=2*2, dilation=2)
            self.conv6_depth = nn.Conv2d(mid_channels, mid_channels, kernel_size=5, groups=mid_channels, padding_mode='circular', padding=2*2, dilation=2)
            
            self.conv_aspp1 = nn.Conv2d(mid_channels, half_channels, kernel_size, padding=padding*2, padding_mode='circular', dilation=2)
            self.conv_aspp2 = nn.Conv2d(mid_channels, half_channels, kernel_size, padding=padding*4, padding_mode='circular', dilation=4)
            self.conv_aspp3 = nn.Conv2d(mid_channels, half_channels, kernel_size, padding=padding*8, padding_mode='circular', dilation=8)
            self.conv_aspp4 = nn.Conv2d(mid_channels, half_channels, kernel_size, padding=padding*16, padding_mode='circular', dilation=16)
            
            self.conv7 = nn.Conv2d(half_channels*4, mid_channels, kernel_size=5, padding=2, padding_mode='circular')
            self.conv8 = nn.Conv2d(mid_channels, mid_channels, kernel_size=5, padding=2, padding_mode='circular')
            self.conv9 = nn.Conv2d(mid_channels, out_channels, 1)
            
        elif dim == 3:
            self.conv1 = nn.Conv3d(in_channels,  mid_channels, kernel_size=5, padding=2, padding_mode='circular')
            self.conv2 = nn.Conv3d(mid_channels, mid_channels, kernel_size=5, padding=2, padding_mode='circular')
            
            self.conv3_point = nn.Conv3d(mid_channels, mid_channels, 1)
            self.conv4_point = nn.Conv3d(mid_channels, mid_channels, 1)
            self.conv5_point = nn.Conv3d(mid_channels, mid_channels, 1)
            self.conv6_point = nn.Conv3d(mid_channels, mid_channels, 1)
            
            self.conv3_depth = nn.Conv3d(mid_channels, mid_channels, kernel_size=5, groups=mid_channels, padding_mode='circular', padding=2*2, dilation=2)
            self.conv4_depth = nn.Conv3d(mid_channels, mid_channels, kernel_size=5, groups=mid_channels, padding_mode='circular', padding=2*2, dilation=2)
            self.conv5_depth = nn.Conv3d(mid_channels, mid_channels, kernel_size=5, groups=mid_channels, padding_mode='circular', padding=2*2, dilation=2)
            self.conv6_depth = nn.Conv3d(mid_channels, mid_channels, kernel_size=5, groups=mid_channels, padding_mode='circular', padding=2*2, dilation=2)
            
            self.conv_aspp1 = nn.Conv3d(mid_channels, half_channels, kernel_size, padding=padding*2, padding_mode='circular', dilation=2)
            self.conv_aspp2 = nn.Conv3d(mid_channels, half_channels, kernel_size, padding=padding*4, padding_mode='circular', dilation=4)
            self.conv_aspp3 = nn.Conv3d(mid_channels, half_channels, kernel_size, padding=padding*8, padding_mode='circular', dilation=8)
            self.conv_aspp4 = nn.Conv3d(mid_channels, half_channels, kernel_size, padding=padding*16, padding_mode='circular', dilation=16)
            
            self.conv7 = nn.Conv3d(half_channels*4, mid_channels, kernel_size=5, padding=2, padding_mode='circular')
            self.conv8 = nn.Conv3d(mid_channels, mid_channels, kernel_size=5, padding=2, padding_mode='circular')
            self.conv9 = nn.Conv3d(mid_channels, out_channels, 1)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = F.relu(self.conv3_point(x))
        x = F.relu(self.conv3_depth(x))
        x = F.relu(self.conv4_point(x))
        x = F.relu(self.conv4_depth(x))
        x = F.relu(self.conv5_point(x))
        x = F.relu(self.conv5_depth(x))
        x = F.relu(self.conv6_point(x))
        x = F.relu(self.conv6_depth(x))
        
        x1 = F.relu(self.conv_aspp1(x))
        x2 = F.relu(self.conv_aspp2(x))
        x3 = F.relu(self.conv_aspp3(x))
        x4 = F.relu(self.conv_aspp4(x))

        x = torch.cat([x1, x2, x3, x4],dim=1)
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x  = self.conv9(x)
        return x
