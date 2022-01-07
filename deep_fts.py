import numpy as np
import torch


class DeepFts():
    def __init__(self, model):
        self.model = model
    
    def eval_mode(self):
        self.model = self.model.half().cuda()

    def train_mode(self):
        self.model = self.model.float().cuda()

    def generate_w_plus(self, w_minus, g_plus, nx):
        
        normal_factor = 10.0 # an arbitrary normalization factor for rescaling
        self.model.eval()
        X = np.zeros([1, 3, np.prod(nx)])
        X[0,0,:] = w_minus/normal_factor 
        X[0,1,:] = g_plus
        std_g_plus = np.std(X[0,1,:])
        X[0,1,:] /= std_g_plus
        X[0,2,:] = std_g_plus/normal_factor
        
        X = torch.tensor(np.reshape(X, [1, 3] + list(nx)), dtype=torch.float16).cuda()
        with torch.no_grad():
            output = self.model(X).detach().cpu().numpy()
            w_plus = np.reshape(output.astype(np.float64)*std_g_plus*normal_factor, np.prod(nx))
            return w_plus
