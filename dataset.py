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
        #n_data = 5000
          
        self.nx = nx
        self.dim = nx.size
        self.file_list = file_list
        self.__n_data = n_data

        self.x_shape = [3] + self.nx.tolist()
        self.y_shape = [1] + self.nx.tolist()

        print(f'{data_dir}, n_data {n_data}, X.shape {self.x_shape}')
        print(f'{data_dir}, n_data {n_data}, Y.shape {self.y_shape}')
        
        self.normal_factor = 10.0 # an arbitrary normalization factor for rescaling w_minus and w_plus_diff
        
        if self.dim == 1:
            self.__X = np.zeros([n_data, 3, np.prod(self.nx)])
            self.__Y = np.zeros([n_data, 1, np.prod(self.nx)])
            for i in range(0, n_data):
                data = np.load(self.file_list[i])
                # exchange field
                self.__X[i,0,:] = data["w_minus"]/self.normal_factor
                # incompressible error
                self.__X[i,1,:] = data["g_plus"]
                # pressure field_diff
                self.__Y[i,0,:] = data["w_plus_diff"]
                
                # normalization
                std_g_plus = np.std(self.__X[i,1,:])
                self.__X[i,1,:] /= std_g_plus
                self.__X[i,2,:] = np.log(std_g_plus)
                self.__Y[i,0,:] /= std_g_plus*self.normal_factor
        
    def __len__(self):
        return self.__n_data
    
    def __getitem__(self, i):
        
        if self.dim == 1:
             return {
                'input' : torch.tensor(np.reshape(self.__X[i,:,:], self.x_shape), dtype=torch.float32),
                'target': torch.tensor(np.reshape(self.__Y[i,:,:], self.y_shape), dtype=torch.float32)
            }
        else:
            X = np.zeros([3, np.prod(self.nx)])
            Y = np.zeros([1, np.prod(self.nx)])
            
            data = np.load(self.file_list[i])
            # exchange field
            X[0,:] = data["w_minus"]/self.normal_factor
            # incompressible error
            X[1,:] = data["g_plus"]
            # pressure field_diff
            Y[0,:] = data["w_plus_diff"]
            # normalization
            std_g_plus = np.std(X[1,:])
            X[1,:] /= std_g_plus
            X[2,:] = np.log(std_g_plus)
            Y[0,:] /= std_g_plus*self.normal_factor
            
            return {
                'input' : torch.tensor(np.reshape(X, self.x_shape), dtype=torch.float32),
                'target': torch.tensor(np.reshape(Y, self.y_shape), dtype=torch.float32)
            }
