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
    def __init__(self, in_channels=1, mid_channels=2, out_channels=1, kernel_size = 3, lr=None, epoch_offset=None):
        super().__init__()
        padding = (kernel_size-1)//2
        self.kernel_size = kernel_size
        self.half_kernel_size = (kernel_size+1)//2
        
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

        # x = torch.arange(self.kernel_size)
        # xv, yv, zv = torch.meshgrid(x, x, x, indexing='ij')
        
        # self.indep_rotation_params = torch.logical_and(torch.logical_and(
        #     xv <= yv,
        #     xv <= zv),
        #     yv <= zv)
        
        # self.indep_flip_params = torch.logical_and(torch.logical_and(
        #     xv < self.half_kernel_size, 
        #     yv < self.half_kernel_size), 
        #     zv < self.half_kernel_size)
        
        # self.indep_params = torch.flatten(torch.logical_and(self.indep_rotation_params, self.indep_flip_params))
        # self.n_indep_params = torch.sum(self.indep_params)
        
        # print(self.indep_params)
        # print(self.n_indep_params)

        assert(kernel_size == 3), \
            "The size of kernel must be 3"

        # (0,1,0)
        # (1,2,1)
        # (0,1,0)

        # (1,2,1)
        # (2,3,2)
        # (1,2,1)
        
        # (0,1,0)
        # (1,2,1)
        # (0,1,0)

        self.n_indep_params = 4
        self.param_mapping_positions = []
        self.param_mapping_positions.append(torch.tensor(
            [[[True, False, True], [False, False, False], [True, False, True]],
           [[False, False, False], [False, False, False], [False, False, False]],
             [[True, False, True], [False, False, False], [True, False, True]]
            ]))

        self.param_mapping_positions.append(torch.tensor(
            [[[False, True, False], [True, False, True], [False, True, False]],
             [[True, False, True], [False, False, False], [True, False, True]],
             [[False, True, False], [True, False, True], [False, True, False]]
            ]))
        
        self.param_mapping_positions.append(torch.tensor(
            [[[False, False, False], [False, True, False], [False, False, False]],
             [[False, True, False],  [True, False, True],  [False, True, False]],
             [[False, False, False], [False, True, False], [False, False, False]]
            ]))

        self.param_mapping_positions.append(torch.tensor(
            [[[False, False, False], [False, False, False], [False, False, False]],
             [[False, False, False], [False, True, False],  [False, False, False]],
             [[False, False, False], [False, False, False], [False, False, False]]
            ]))
        
        self.weight1   = torch.nn.Parameter(torch.randn([mid_channels,  in_channels, self.n_indep_params]), requires_grad=True)
        self.weight2   = torch.nn.Parameter(torch.randn([mid_channels, mid_channels, self.n_indep_params]), requires_grad=True)
        self.weight3   = torch.nn.Parameter(torch.randn([mid_channels, mid_channels, self.n_indep_params]), requires_grad=True)
        self.weight4_1 = torch.nn.Parameter(torch.randn([mid_channels, mid_channels, self.n_indep_params]), requires_grad=True)
        self.weight4_2 = torch.nn.Parameter(torch.randn([mid_channels, mid_channels, self.n_indep_params]), requires_grad=True)
        self.weight5   = torch.nn.Parameter(torch.randn([mid_channels, mid_channels, self.n_indep_params]), requires_grad=True)
        self.weight6   = torch.nn.Parameter(torch.randn([mid_channels, mid_channels, self.n_indep_params]), requires_grad=True)
        self.conv7   = nn.Conv3d(mid_channels, out_channels, 1)

        self.bm1   = nn.BatchNorm3d(mid_channels)
        self.bm2   = nn.BatchNorm3d(mid_channels)
        self.bm3   = nn.BatchNorm3d(mid_channels)
        self.bm4_1 = nn.BatchNorm3d(mid_channels)
        self.bm4_2 = nn.BatchNorm3d(mid_channels)
        self.bm5   = nn.BatchNorm3d(mid_channels)
        self.bm6   = nn.BatchNorm3d(mid_channels)
        
        nn.init.kaiming_uniform_(self.weight1)
        nn.init.kaiming_uniform_(self.weight2)
        nn.init.kaiming_uniform_(self.weight3)
        nn.init.kaiming_uniform_(self.weight4_1)
        nn.init.kaiming_uniform_(self.weight4_2)
        nn.init.kaiming_uniform_(self.weight5)
        nn.init.kaiming_uniform_(self.weight6)
        
    def make_equiv_weight(self, input_weight):

        weight = F.pad(input_weight, (0,self.kernel_size*self.kernel_size*self.kernel_size-self.n_indep_params), "constant", 0)
        weight = torch.reshape(weight, (input_weight.shape[0], input_weight.shape[1], self.kernel_size, self.kernel_size, self.kernel_size))
        
        for i in range(self.n_indep_params):
            weight[:,:,self.param_mapping_positions[i]] = torch.reshape(input_weight[:,:,], (input_weight.shape[0], input_weight.shape[1], input_weight.shape[2], 1))[:,:,i,:]

        return weight
    
    def forward(self, x):
        
        conv_weight1   = self.make_equiv_weight(self.weight1)
        conv_weight2   = self.make_equiv_weight(self.weight2)
        conv_weight3   = self.make_equiv_weight(self.weight3)
        conv_weight4_1 = self.make_equiv_weight(self.weight4_1)
        conv_weight4_2 = self.make_equiv_weight(self.weight4_2)
        conv_weight5   = self.make_equiv_weight(self.weight5)
        conv_weight6   = self.make_equiv_weight(self.weight6)
        
        x = F.mish(self.bm1  (F.conv3d(F.pad(x, self.padding,   'circular'), conv_weight1)))
        x = F.mish(self.bm2  (F.conv3d(F.pad(x, self.padding,   'circular'), conv_weight2)))
        x = F.mish(self.bm3  (F.conv3d(F.pad(x, self.padding_2, 'circular'), conv_weight3,   dilation=2)))
        x = F.mish(self.bm4_1(F.conv3d(F.pad(x, self.padding_4, 'circular'), conv_weight4_1, dilation=4)))
        x = F.mish(self.bm4_2(F.conv3d(F.pad(x, self.padding_8, 'circular'), conv_weight4_2, dilation=8)))
        x = F.mish(self.bm5  (F.conv3d(F.pad(x, self.padding,   'circular'), conv_weight5)))
        x = F.mish(self.bm6  (F.conv3d(F.pad(x, self.padding,   'circular'), conv_weight6)))
        x = self.conv7(x)
        
        return x

nx = [64, 64, 64]
net = NeuralNetwork()

X = []
Y = []

X.append(torch.rand(nx))
print(X[0].shape)

#------------------ Flip ----------------------
flip_dim_1 = (0, 1)
flip_dim_2 = (2, 1)
flip_dim_3 = (2, 0, 1)

X.append(torch.flip(X[0].clone(), dims=flip_dim_1))
X.append(torch.flip(X[0].clone(), dims=flip_dim_2))
X.append(torch.flip(X[0].clone(), dims=flip_dim_3))

#
axis = [0, 1]
X.append(torch.rot90(X[0].clone(), 1, axis))
X.append(torch.rot90(X[0].clone(), 2, axis))
X.append(torch.rot90(X[0].clone(), 3, axis))

# net(torch.reshape(X[0], (1, 1, nx[0], nx[1], nx[2])))
for i in range(0,len(X)):
    X[i] = torch.reshape(X[i], (1, 1, nx[0], nx[1], nx[2]))
    Y.append(torch.reshape(net(X[i]), (nx[0], nx[1], nx[2])))
    print(X[i].shape)
    print(Y[i].shape)

print(torch.std(Y[1] - torch.flip(Y[0].clone(), dims=flip_dim_1)))
print(torch.std(Y[2] - torch.flip(Y[0].clone(), dims=flip_dim_2)))
print(torch.std(Y[3] - torch.flip(Y[0].clone(), dims=flip_dim_3)))

# print(Y[4])
# print(Y[5])
# print(Y[6])

axis = [0, 1]
print(torch.std(Y[4] - torch.rot90(Y[0].clone(), 1, axis)))
print(torch.std(Y[5] - torch.rot90(Y[0].clone(), 2, axis)))
print(torch.std(Y[6] - torch.rot90(Y[0].clone(), 3, axis)))


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