import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl

class NeuralNetwork(pl.LightningModule):
    def __init__(self, in_channels=1, mid_channels=1, out_channels=1, kernel_size = 3, lr=None, epoch_offset=None):
        super().__init__()
        padding = (kernel_size-1)//2
        half_kernel_size = (kernel_size+1)//2
        self.padding = (padding,)*6
        self.padding_2 = tuple([2*p for p in self.padding])
        self.padding_4 = tuple([4*p for p in self.padding])
        self.padding_8 = tuple([8*p for p in self.padding])
        
        # print(self.padding)
        # print(self.padding_2)
        # print(self.padding_4)
        # print(self.padding_8)
        
        self.loss = torch.nn.MSELoss()
        
        if lr:
            self.lr = lr
        else:
            self.lr = 1e-3

        if epoch_offset:
            self.epoch_offset = epoch_offset
        else:
            self.epoch_offset = 0

        self.weight_conv1   = torch.randn([mid_channels,  in_channels, half_kernel_size,half_kernel_size,half_kernel_size])
        self.weight_conv2   = torch.randn([mid_channels, mid_channels, half_kernel_size,half_kernel_size,half_kernel_size])
        self.weight_conv3   = torch.randn([mid_channels, mid_channels, half_kernel_size,half_kernel_size,half_kernel_size])
        self.weight_conv4_1 = torch.randn([mid_channels, mid_channels, half_kernel_size,half_kernel_size,half_kernel_size])
        self.weight_conv4_2 = torch.randn([mid_channels, mid_channels, half_kernel_size,half_kernel_size,half_kernel_size])
        self.weight_conv5   = torch.randn([mid_channels, mid_channels, half_kernel_size,half_kernel_size,half_kernel_size])
        self.weight_conv6   = torch.randn([mid_channels, mid_channels, half_kernel_size,half_kernel_size,half_kernel_size])
        self.weight_conv7   = torch.randn([out_channels, mid_channels, 1,1,1])
        
        # print(self.weight_conv1.shape)
        # print(self.weight_conv2.shape)
        # print(self.weight_conv7.shape)

        self.bm1   = nn.BatchNorm3d(mid_channels)
        self.bm2   = nn.BatchNorm3d(mid_channels)
        self.bm3   = nn.BatchNorm3d(mid_channels)
        self.bm4_1 = nn.BatchNorm3d(mid_channels)
        self.bm4_2 = nn.BatchNorm3d(mid_channels)
        self.bm5   = nn.BatchNorm3d(mid_channels)
        self.bm6   = nn.BatchNorm3d(mid_channels)
        
    def make_equiv_weight(self, input_weight, padding):
    
        weight = F.pad(input_weight, (0,padding,0,padding,0,padding), "constant", 0)
        weight = weight + torch.flip(weight, dims=(2,))
        weight = weight + torch.flip(weight, dims=(3,))
        weight = weight + torch.flip(weight, dims=(4,))

        return weight
    
    def forward(self, x):
        
        weight_conv1   = self.make_equiv_weight(self.weight_conv1,   self.padding[0])
        weight_conv2   = self.make_equiv_weight(self.weight_conv2,   self.padding[0])
        weight_conv3   = self.make_equiv_weight(self.weight_conv3,   self.padding[0])
        weight_conv4_1 = self.make_equiv_weight(self.weight_conv4_1, self.padding[0])
        weight_conv4_2 = self.make_equiv_weight(self.weight_conv4_2, self.padding[0])
        weight_conv5   = self.make_equiv_weight(self.weight_conv5,   self.padding[0])
        weight_conv6   = self.make_equiv_weight(self.weight_conv6,   self.padding[0])
        
        x = F.mish(self.bm1  (F.conv3d(F.pad(x, self.padding,   'circular'), weight_conv1)))
        x = F.mish(self.bm2  (F.conv3d(F.pad(x, self.padding,   'circular'), weight_conv2)))
        x = F.mish(self.bm3  (F.conv3d(F.pad(x, self.padding_2, 'circular'), weight_conv3,   dilation=2)))
        x = F.mish(self.bm4_1(F.conv3d(F.pad(x, self.padding_4, 'circular'), weight_conv4_1, dilation=4)))
        x = F.mish(self.bm4_2(F.conv3d(F.pad(x, self.padding_8, 'circular'), weight_conv4_2, dilation=8)))
        x = F.mish(self.bm5  (F.conv3d(F.pad(x, self.padding,   'circular'), weight_conv5)))
        x = F.mish(self.bm6  (F.conv3d(F.pad(x, self.padding,   'circular'), weight_conv6)))
        x = F.conv3d(x, self.weight_conv7)
        
        return x

nx = [64, 64, 64]
X_0 = torch.rand(nx)
print(X_0.shape)

#------------------ Flip ----------------------
flip_dim_1 = (0, 1)
flip_dim_2 = (2, 1)
flip_dim_3 = (2, 0, 1)

X_1 = torch.flip(X_0.clone(), dims=flip_dim_1)
X_2 = torch.flip(X_0.clone(), dims=flip_dim_2)
X_3 = torch.flip(X_0.clone(), dims=flip_dim_3)

# print(X_0)
# print(X_1)
# print(X_2)

X_0 = torch.reshape(X_0, (1, 1, nx[0], nx[1], nx[2]))
X_1 = torch.reshape(X_1, (1, 1, nx[0], nx[1], nx[2]))
X_2 = torch.reshape(X_2, (1, 1, nx[0], nx[1], nx[2]))
X_3 = torch.reshape(X_3, (1, 1, nx[0], nx[1], nx[2]))

# print(X_0.shape)
# print(X_1.shape)
# print(X_2.shape)
# print(X_3.shape)

net = NeuralNetwork()
Y_0 = torch.reshape(net(X_0), (nx[0], nx[1], nx[2]))
Y_1 = torch.reshape(net(X_1), (nx[0], nx[1], nx[2]))
Y_2 = torch.reshape(net(X_2), (nx[0], nx[1], nx[2]))
Y_3 = torch.reshape(net(X_3), (nx[0], nx[1], nx[2]))

# print(Y_0[:,:,0])
# print(Y_1[:,:,0])
# print(Y_2[:,:,0])

print(torch.std(Y_1 - torch.flip(Y_0.clone(), dims=flip_dim_1)))
print(torch.std(Y_2 - torch.flip(Y_0.clone(), dims=flip_dim_2)))
print(torch.std(Y_3 - torch.flip(Y_0.clone(), dims=flip_dim_3)))

# def equiv_conv(self, conv, x):
#     x_0 = conv(x)
    
#     axis = [2, 3]
#     xy1 = torch.rot90(conv(torch.rot90(x, 1, axis)), 3, axis)
#     xy2 = torch.rot90(conv(torch.rot90(x, 2, axis)), 2, axis)
#     xy3 = torch.rot90(conv(torch.rot90(x, 3, axis)), 1, axis)
    
#     axis = [2, 4]
#     xz1 = torch.rot90(conv(torch.rot90(x, 1, axis)), 3, axis)
#     xz2 = torch.rot90(conv(torch.rot90(x, 2, axis)), 2, axis)
#     xz3 = torch.rot90(conv(torch.rot90(x, 3, axis)), 1, axis)
    
#     axis = [3, 4]
#     yz1 = torch.rot90(conv(torch.rot90(x, 1, axis)), 3, axis)
#     yz2 = torch.rot90(conv(torch.rot90(x, 2, axis)), 2, axis)
#     yz3 = torch.rot90(conv(torch.rot90(x, 3, axis)), 1, axis)
    
#     out = x_0 + xy1 + xy2 + xy3 + xz1 + xz2 + xz3 + yz1 + yz2 + yz3
    
#     out = torch.flip(out, dims=(2,)) + out
#     out = torch.flip(out, dims=(3,)) + out
#     out = torch.flip(out, dims=(4,)) + out
    
#     return out
    
#     x = F.mish(self.bm1  (self.equiv_conv(self.conv1,  x)))
#     x = F.mish(self.bm2  (self.equiv_conv(self.conv2,  x)))
#     x = F.mish(self.bm3  (self.equiv_conv(self.conv3,  x)))
#     # x = F.mish(self.bm4_1(self.equiv_conv(self.conv4_1,x)))
#     # x = F.mish(self.bm4_2(self.equiv_conv(self.conv4_2,x)))
#     x = F.mish(self.bm5  (self.equiv_conv(self.conv5,  x)))
#     x = F.mish(self.bm6  (self.equiv_conv(self.conv6,  x)))
#     x =                   self.equiv_conv(self.conv7,  x)
    
#     X_1 = torch.rot90(X_0.clone(), 1, [0, 1])
#     X_2 = torch.rot90(X_0.clone(), 2, [0, 1])
#     X_3 = torch.rot90(X_0.clone(), 3, [0, 1])

#     # print(X_0)
#     # print(X_1)
#     # print(X_2)

#     X_0 = torch.reshape(X_0, (1, 1, nx[0], nx[1], nx[2]))
#     X_1 = torch.reshape(X_1, (1, 1, nx[0], nx[1], nx[2]))
#     X_2 = torch.reshape(X_2, (1, 1, nx[0], nx[1], nx[2]))
#     X_3 = torch.reshape(X_3, (1, 1, nx[0], nx[1], nx[2]))

#     print(X_0.shape)
#     print(X_1.shape)
#     print(X_2.shape)
#     print(X_3.shape)

#     net = NeuralNetwork()
#     Y_0 = torch.reshape(net(X_0), (nx[0], nx[1], nx[2]))
#     Y_1 = torch.reshape(net(X_1), (nx[0], nx[1], nx[2]))
#     Y_2 = torch.reshape(net(X_2), (nx[0], nx[1], nx[2]))
#     Y_3 = torch.reshape(net(X_3), (nx[0], nx[1], nx[2]))

#     print(Y_0[:,:,0])
#     print(Y_1[:,:,0])
#     print(Y_2[:,:,0])

#     print(torch.std(Y_1 - torch.rot90(Y_0.clone(), 1, [0, 1])))
#     print(torch.std(Y_2 - torch.rot90(Y_0.clone(), 2, [0, 1])))
#     print(torch.std(Y_3 - torch.rot90(Y_0.clone(), 3, [0, 1])))