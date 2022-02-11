import numpy as np
import glob
import logging
import torch

class FtsDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir):

        file_list = glob.glob(data_dir + "/*.npz")
        file_list.sort()
        sample_data = np.load(file_list[0])
        nx = sample_data["nx"]
        n_data = len(file_list)
          
        self.nx = nx
        self.dim = nx.size
        self.file_list = file_list
        self.__n_data = n_data

        self.x_shape = [3] + self.nx.tolist()
        self.y_shape = [1] + self.nx.tolist()

        print(f'{data_dir}, n_data {n_data}, X.shape {self.x_shape}')
        print(f'{data_dir}, n_data {n_data}, Y.shape {self.y_shape}')
        
    def __len__(self):
        return self.__n_data
    
    def __getitem__(self, i):
        
        X = np.zeros([3, np.prod(self.nx)])
        Y = np.zeros([1, np.prod(self.nx)])
                    
        data = np.load(self.file_list[i])
        # exchange field
        X[0,:] = data["w_minus"]/10
        # incompressible error
        X[1,:] = data["g_plus"]
        # pressure field_diff
        Y[0,:] = data["w_plus_diff"]
        # normalization
        std_g_plus = np.std(X[1,:])
        X[1,:] /= std_g_plus
        X[2,:] = np.sqrt(std_g_plus)
        Y[0,:] /= std_g_plus*20

        #flip
        X = np.reshape(X, self.x_shape)
        Y = np.reshape(Y, self.y_shape)
        
        if(self.dim >= 3 and np.random.choice([True,False])):
            X = np.flip(X, 3)
            Y = np.flip(Y, 3)
        if(self.dim >= 2 and np.random.choice([True,False])):
            X = np.flip(X, 2)
            Y = np.flip(Y, 2)
        if(self.dim >= 1 and np.random.choice([True,False])):
            X = np.flip(X, 1)
            Y = np.flip(Y, 1)
        
        X = torch.from_numpy(X.copy())
        Y = torch.from_numpy(Y.copy())
        
        return {
            'input' : X.to(dtype=torch.float32),
            'target': Y.to(dtype=torch.float32)
        }
