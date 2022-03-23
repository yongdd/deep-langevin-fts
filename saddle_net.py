import numpy as np
import torch

from model.unet import *
from model.aspp import *
from model.atrpar import *
from model.atrcas import *
from model.atrcasx import *

class SaddleNet(LitAtrPar): # LitUNet, LitASPP, LitAtrPar, LitAtrCas, LitAtrCasX
    def __init__(self, dim, mid_channels):
        super().__init__(dim=dim, mid_channels=mid_channels)
        self.eval()
        self.half().cuda()

    def predict_w_plus(self, w_minus, g_plus, nx):
        
        X = np.zeros([1, 3, np.prod(nx)])
        X[0,0,:] = w_minus/10.0
        X[0,1,:] = g_plus
        
        # zero mean
        X[0,0,:] -= np.mean(X[0,0,:])
        X[0,1,:] -= np.mean(X[0,1,:])
        
        # normalization
        std_g_plus = np.std(X[0,1,:])
        X[0,1,:] /= std_g_plus
        X[0,2,:] = np.sqrt(std_g_plus)

        X = torch.tensor(np.reshape(X, [1, 3] + list(nx)), dtype=torch.float16).cuda()
        with torch.no_grad():
            output = self(X).detach().cpu().numpy()
            w_plus = np.reshape(output.astype(np.float64)*std_g_plus*20, np.prod(nx))
            return w_plus
