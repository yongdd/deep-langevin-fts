import numpy as np
import torch

from model.unet import *             # LitUNet, 
from model.atr_par_ip import *       # LitAtrousParallelImagePooling, 
from model.atr_par_ip_mish import *  # LitAtrousParallelImagePoolingMish, 
from model.atr_par import *          # LitAtrousParallel, 
from model.atr_par_mish import *     # LitAtrousParallelMish, 
from model.atr_cas import *          # LitAtrousCascade, 
from model.atr_cas_mish import *     # LitAtrousCascadeMish, 
from model.atr_cas_x import *        # LitAtrousCascadeXception, 

class InferenceNet(LitAtrousCascadeMish):
    def __init__(self, dim, features):
        super().__init__(dim=dim, mid_channels=features)
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
