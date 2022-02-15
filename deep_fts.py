import numpy as np
import torch


class DeepFts():
    def __init__(self, model):
        self.model = model
    
    def eval_mode(self):
        self.model.eval()
        self.model = self.model.half().cuda()

    def train_mode(self):
        self.model.train()
        self.model = self.model.float().cuda()

    def generate_w_plus(self, w_minus, g_plus, nx):
        
        X = np.zeros([1, 3, np.prod(nx)])
        X[0,0,:] = w_minus/10.0
        X[0,1,:] = g_plus
        std_g_plus = np.std(X[0,1,:])
        X[0,1,:] /= std_g_plus
        X[0,2,:] = np.sqrt(std_g_plus)

        X = torch.tensor(np.reshape(X, [1, 3] + list(nx)), dtype=torch.float16).cuda()
        with torch.no_grad():
            output = self.model(X).detach().cpu().numpy()
            w_plus = np.reshape(output.astype(np.float64)*std_g_plus*20, np.prod(nx))
            return w_plus
