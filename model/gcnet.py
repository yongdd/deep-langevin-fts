import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl

class LitGCNet(pl.LightningModule): # Global Convolutional Network
    def __init__(self, dim, in_channels=3, mid_channels = 32, out_channels=1, kernel_size = 3):
        super().__init__()
        padding = (kernel_size-1)//2
        
        gcn_channels = 21
        g_kernel_size = 9
        g_padding = (g_kernel_size-1)//2
        
        if dim == 3:

            self.conv1 = nn.Conv2d(in_channels,  mid_channels, kernel_size, bias=False, padding=padding, padding_mode='circular')
            self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size, bias=False, padding=padding, padding_mode='circular')
            self.conv3 = nn.Conv2d(mid_channels, mid_channels, kernel_size, bias=False, padding=padding*2, padding_mode='circular', dilation=2)

            self.conv4_1_1 = nn.Conv2d(mid_channels, mid_channels, kernel_size=(g_kernel_size,1,1), padding_mode='circular', padding=(g_padding,0,0))
            self.conv4_1_2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=(1,g_kernel_size,1), padding_mode='circular', padding=(0,g_padding,0))
            self.conv4_1_3 = nn.Conv2d(mid_channels, mid_channels, kernel_size=(1,1,g_kernel_size), padding_mode='circular', padding=(0,0,g_padding))
            
            self.conv4_1_12 = nn.Conv2d(mid_channels, mid_channels, kernel_size=(1,g_kernel_size,1), padding_mode='circular', padding=(0,g_padding,0))
            self.conv4_1_13 = nn.Conv2d(mid_channels, mid_channels, kernel_size=(1,1,g_kernel_size), padding_mode='circular', padding=(0,0,g_padding))
            self.conv4_1_21 = nn.Conv2d(mid_channels, mid_channels, kernel_size=(g_kernel_size,1,1), padding_mode='circular', padding=(g_padding,0,0))
            self.conv4_1_23 = nn.Conv2d(mid_channels, mid_channels, kernel_size=(1,1,g_kernel_size), padding_mode='circular', padding=(0,0,g_padding))
            self.conv4_1_31 = nn.Conv2d(mid_channels, mid_channels, kernel_size=(g_kernel_size,1,1), padding_mode='circular', padding=(g_padding,0,0))
            self.conv4_1_32 = nn.Conv2d(mid_channels, mid_channels, kernel_size=(1,g_kernel_size,1), padding_mode='circular', padding=(0,g_padding,0))

            self.conv4_1_123 = nn.Conv2d(mid_channels, mid_channels, kernel_size=(1,1,g_kernel_size), padding_mode='circular', padding=(0,0,g_padding))
            self.conv4_1_132 = nn.Conv2d(mid_channels, mid_channels, kernel_size=(1,g_kernel_size,1), padding_mode='circular', padding=(0,g_padding,0))
            self.conv4_1_213 = nn.Conv2d(mid_channels, mid_channels, kernel_size=(1,1,g_kernel_size), padding_mode='circular', padding=(0,0,g_padding))
            self.conv4_1_231 = nn.Conv2d(mid_channels, mid_channels, kernel_size=(g_kernel_size,1,1), padding_mode='circular', padding=(g_padding,0,0))
            self.conv4_1_312 = nn.Conv2d(mid_channels, mid_channels, kernel_size=(1,g_kernel_size,1), padding_mode='circular', padding=(0,g_padding,0))
            self.conv4_1_321 = nn.Conv2d(mid_channels, mid_channels, kernel_size=(g_kernel_size,1,1), padding_mode='circular', padding=(g_padding,0,0))

            self.conv4_1_231 = nn.Conv2d(mid_channels, mid_channels, kernel_size=(1,g_kernel_size,1), padding_mode='circular', padding=(0,g_padding,0))
            self.conv4_1_312 = nn.Conv2d(mid_channels, mid_channels, kernel_size=(1,g_kernel_size,1), padding_mode='circular', padding=(0,g_padding,0))
            self.conv4_1_321 = nn.Conv2d(mid_channels, mid_channels, kernel_size=(1,g_kernel_size,1), padding_mode='circular', padding=(0,g_padding,0))        
            
            self.conv4_1_132 = nn.Conv2d(mid_channels, mid_channels, kernel_size=(1,g_kernel_size,1), padding_mode='circular', padding=(0,g_padding,0))
            
            self.conv4_1_12 = nn.Conv2d(mid_channels, gcn_channels, kernel_size=(g_kernel_size,1), padding_mode='circular', padding=(g_padding,0))
            self.conv4_1_21 = nn.Conv2d(mid_channels, mid_channels, kernel_size=(g_kernel_size,1), padding_mode='circular', padding=(g_padding,0))
            self.conv4_1_22 = nn.Conv2d(mid_channels, gcn_channels, kernel_size=(1,g_kernel_size), padding_mode='circular', padding=(0,g_padding))

            self.conv4_2_11 = nn.Conv2d(mid_channels, mid_channels, kernel_size=(1,g_kernel_size), padding_mode='circular', padding=(0,g_padding*2),  dilation=2)
            self.conv4_2_12 = nn.Conv2d(mid_channels, gcn_channels, kernel_size=(g_kernel_size,1), padding_mode='circular', padding=(g_padding*2,0),  dilation=2)
            self.conv4_2_21 = nn.Conv2d(mid_channels, mid_channels, kernel_size=(g_kernel_size,1), padding_mode='circular', padding=(g_padding*2,0),  dilation=2)
            self.conv4_2_22 = nn.Conv2d(mid_channels, gcn_channels, kernel_size=(1,g_kernel_size), padding_mode='circular', padding=(0,g_padding*2),  dilation=2)
              
            self.conv4_3_11 = nn.Conv2d(mid_channels, mid_channels, kernel_size=(1,g_kernel_size), padding_mode='circular', padding=(0,g_padding*4),  dilation=4)
            self.conv4_3_12 = nn.Conv2d(mid_channels, gcn_channels, kernel_size=(g_kernel_size,1), padding_mode='circular', padding=(g_padding*4,0),  dilation=4)
            self.conv4_3_21 = nn.Conv2d(mid_channels, mid_channels, kernel_size=(g_kernel_size,1), padding_mode='circular', padding=(g_padding*4,0),  dilation=4)
            self.conv4_3_22 = nn.Conv2d(mid_channels, gcn_channels, kernel_size=(1,g_kernel_size), padding_mode='circular', padding=(0,g_padding*4),  dilation=4)
              
            self.conv5 = nn.Conv2d(mid_channels+gcn_channels*3, mid_channels, kernel_size, bias=False, padding=padding, padding_mode='circular')
            self.conv6 = nn.Conv2d(mid_channels, mid_channels, kernel_size, bias=False, padding=padding, padding_mode='circular')
            self.conv7 = nn.Conv2d(mid_channels, out_channels, 1)
            
            self.bm1   = nn.BatchNorm2d(mid_channels)
            self.bm2   = nn.BatchNorm2d(mid_channels)
            self.bm3   = nn.BatchNorm2d(mid_channels)
            self.bm4_1 = nn.BatchNorm2d(gcn_channels)
            self.bm4_2 = nn.BatchNorm2d(gcn_channels)
            self.bm4_3 = nn.BatchNorm2d(gcn_channels)
            self.bm5   = nn.BatchNorm2d(mid_channels)
            self.bm6   = nn.BatchNorm2d(mid_channels)

    def forward(self, x):
        x = F.relu(self.bm1(self.conv1(x)))
        x = F.relu(self.bm2(self.conv2(x)))
        x = F.relu(self.bm3(self.conv3(x)))

        x11 = self.conv4_1_11(x)
        x11 = self.conv4_1_12(x11)
        x12 = self.conv4_1_21(x)
        x12 = self.conv4_1_22(x12)
        x1  = F.relu(self.bm4_1(x11 + x12))

        x21 = self.conv4_2_11(x)
        x21 = self.conv4_2_12(x21)
        x22 = self.conv4_2_21(x)
        x22 = self.conv4_2_22(x22)
        x2  = F.relu(self.bm4_2(x21 + x22))
        
        x31 = self.conv4_3_11(x)
        x31 = self.conv4_3_12(x31)
        x32 = self.conv4_3_21(x)
        x32 = self.conv4_3_22(x32)
        x3  = F.relu(self.bm4_3(x31 + x32))

        x = torch.cat([x, x1, x2, x3],dim=1)
        x = F.relu(self.bm5(self.conv5(x)))
        x = F.relu(self.bm6(self.conv6(x)))
        x = self.conv7(x)
        return x
