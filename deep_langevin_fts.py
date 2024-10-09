import os
import sys
import time
import re
import glob
import shutil
import pathlib
import copy
import numpy as np
import itertools
import networkx as nx
import matplotlib.pyplot as plt
from scipy.io import savemat, loadmat
from langevinfts import *

import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.strategies.ddp import DDPStrategy

# from model.unet import *             # LitUNet, 
# from model.atr_par_ip import *       # LitAtrousParallelImagePooling, 
# from model.atr_par_ip_mish import *  # LitAtrousParallelImagePoolingMish, 
# from model.atr_par import *          # LitAtrousParallel, 
# from model.atr_par_mish import *     # LitAtrousParallelMish, 
# from model.atr_cas import *          # LitAtrousCascade, 
# from model.atr_cas_mish import *     # LitAtrousCascadeMish, 
# from model.atr_cas_x import *        # LitAtrousCascadeXception, 

# OpenMP environment variables
os.environ["MKL_NUM_THREADS"] = "1"  # always 1
os.environ["OMP_STACKSIZE"] = "1G"
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"  # 0, 1
os.environ["OMP_NUM_THREADS"] = "2"  # 1 ~ 4

class TrainAndInference(pl.LightningModule):
    def __init__(self, dim, in_channels=3, mid_channels=32, out_channels=1, kernel_size=3, lr=None, epoch_offset=None):
        super().__init__()
        padding = (kernel_size-1)//2
        self.dim = dim
        self.loss = torch.nn.MSELoss()
        
        if lr:
            self.lr = lr
        else:
            self.lr = 1e-3

        if epoch_offset:
            self.epoch_offset = epoch_offset
        else:
            self.epoch_offset = 0

        if dim == 3:
            self.conv1   = nn.Conv3d(in_channels,  mid_channels, kernel_size, bias=False, padding=padding, padding_mode='circular')
            self.conv2   = nn.Conv3d(mid_channels, mid_channels, kernel_size, bias=False, padding=padding, padding_mode='circular')
            self.conv3   = nn.Conv3d(mid_channels, mid_channels, kernel_size, bias=False, padding=padding*2, padding_mode='circular', dilation=2)
            self.conv4_1 = nn.Conv3d(mid_channels, mid_channels, kernel_size, bias=False, padding=padding*4, padding_mode='circular', dilation=4)
            self.conv4_2 = nn.Conv3d(mid_channels, mid_channels, kernel_size, bias=False, padding=padding*8, padding_mode='circular', dilation=8)
            self.conv5   = nn.Conv3d(mid_channels, mid_channels, kernel_size, bias=False, padding=padding, padding_mode='circular')
            self.conv6   = nn.Conv3d(mid_channels, mid_channels, kernel_size, bias=False, padding=padding, padding_mode='circular')
            self.conv7   = nn.Conv3d(mid_channels, out_channels, 1)

            self.bm1   = nn.BatchNorm3d(mid_channels)
            self.bm2   = nn.BatchNorm3d(mid_channels)
            self.bm3   = nn.BatchNorm3d(mid_channels)
            self.bm4_1 = nn.BatchNorm3d(mid_channels)
            self.bm4_2 = nn.BatchNorm3d(mid_channels)
            self.bm5   = nn.BatchNorm3d(mid_channels)
            self.bm6   = nn.BatchNorm3d(mid_channels)

    def forward(self, x):
        x = F.mish(self.bm1(self.conv1(x)))
        x = F.mish(self.bm2(self.conv2(x)))
        x = F.mish(self.bm3(self.conv3(x)))
        x = F.mish(self.bm4_1(self.conv4_1(x)))
        x = F.mish(self.bm4_2(self.conv4_2(x)))
        x = F.mish(self.bm5(self.conv5(x)))
        x = F.mish(self.bm6(self.conv6(x)))
        x = self.conv7(x)
        return x

    def set_normalization_factor(self, normal_factor):
        self.normalization_factor = normal_factor

    def set_inference_mode(self, device):
        self.eval()
        self.half().to(device)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[100], gamma=0.2,
            verbose=False)
        return [optimizer], [scheduler]
    
    def on_train_start(self):
        total_params = sum(p.numel() for p in self.parameters())
        self.log('total_params', float(total_params), sync_dist=True)
        #print("total_params", total_params)
    
    def on_train_epoch_start(self):
        self.log('learning_rate', self.optimizers().param_groups[0]['lr'], sync_dist=True)
        #print('\n')

    def on_train_epoch_end(self):
        path = "saved_model_weights"
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        torch.save([self.state_dict(), self.normalization_factor], os.path.join(path, 'epoch_%d.pth' % (self.current_epoch + self.epoch_offset)))
      
    def training_step(self, train_batch, batch_idx):
        x = train_batch['input']
        y = train_batch['target']
        x = self(x)   
        loss = self.loss(y, x)
        self.log('train_loss', loss, sync_dist=True)
        return loss

    def predict_w_imag(self, w_real, h_deriv, nx):
    
        # The numbers of real and imaginary fields, respectively
        R = len(w_real)
        I = len(h_deriv)
    
        # The number of input channels
        in_channels = R+2*I

        # Copy field arrays into gpu memory
        d_w_real = torch.tensor(w_real, dtype=torch.float64).to(self.device)
        d_h_deriv = torch.tensor(h_deriv, dtype=torch.float64).to(self.device)

        # Make Input Array
        X = torch.zeros([1, in_channels, np.prod(nx)], dtype=torch.float64).to(self.device)

        # For each real field
        for i in range(R):
            X[0,i,:] = (d_w_real[i]-torch.mean(d_w_real[i]))/10.0

        # For each imaginary field
        total_std_h_deriv = []
        for i in range(I):
            std_h_deriv = torch.std(d_h_deriv[i])
            total_std_h_deriv.append(std_h_deriv)
            # Normalization
            X[0,R+2*i  ,:] = (d_h_deriv[i] - torch.mean(d_h_deriv[i]))/std_h_deriv
            # Normalization factor
            X[0,R+2*i+1,:] =  torch.sqrt(std_h_deriv)
        
        X = torch.reshape(X, [1, in_channels] + list(nx)).type(torch.float16)
        
        with torch.no_grad():
            # Neural network prediction
            output = self(X)
            # Rescaling output
            for i in range(I):
                output[0,i,:,:,] *= total_std_h_deriv[i]*self.normalization_factor[i]
            d_w_diff = torch.reshape(output.type(torch.float64), (I,-1,))
            return d_w_diff.detach().cpu().numpy()

class FtsDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, R, I):

        # The numbers of real and imaginary fields, respectively
        self.R = R
        self.I = I

        # The number of input channels
        self.in_channels = R+2*I

        file_list = sorted(glob.glob(data_dir + "/*.npz"))
        sample_data = np.load(file_list[0])
        nx = sample_data["nx"]
        n_data = len(file_list)
          
        self.nx = nx
        self.dim = nx.size
        self.file_list = file_list
        self.__n_data = n_data

        self.x_shape = [self.in_channels] + self.nx.tolist()
        self.y_shape = [self.I]           + self.nx.tolist()

        print(f'{data_dir}, n_data {n_data}, X.shape {self.x_shape}')
        print(f'{data_dir}, n_data {n_data}, Y.shape {self.y_shape}')
        
        # Compute standard deviations of output for normalization
        output_std = np.zeros([self.I, n_data])
        for n in range(n_data):
            data = np.load(file_list[n])
            output_std[:,n] =  data["w_diff_std"]/data["h_deriv_std"]
        self.normalization_factor = np.mean(output_std, axis=1)
        print("Normalization factors: ", self.normalization_factor)

    def get_normalization_factor(self,):
        return self.normalization_factor        

    def __len__(self):
        return self.__n_data
    
    def __getitem__(self, data_idx):
                
        X = np.zeros([self.in_channels, np.prod(self.nx)])
        Y = np.zeros([self.I,           np.prod(self.nx)])
                    
        data = np.load(self.file_list[data_idx])
        
        # For each real field
        for i in range(self.R):
            X[i,:] = (data["w_real"][i]-np.mean(data["w_real"][i]))/10.0

        # For each imaginary field
        for i in range(self.I):
            std_h_deriv = np.std(data["h_deriv"][i].astype(np.float64))
            # Normalization
            X[self.R+2*i  ,:] = (data["h_deriv"][i] - np.mean(data["h_deriv"][i]))/std_h_deriv
            # Normalization factor
            X[self.R+2*i+1,:] =  np.sqrt(std_h_deriv)
            # Pressure field_diff
            Y[i,:] = data["w_diff"][i]
            Y[i,:] /= std_h_deriv*self.normalization_factor[i]

        # Flip
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
        
        # for i in range(self.I):
        #    print(self.file_list[data_idx], i, torch.std(Y[i]))
        
        return {
            'input' : X.to(dtype=torch.float32),
            'target': Y.to(dtype=torch.float32)
        }

# Well-tempered metadynamics
# [*J. Chem. Phys.* **2022**, 157, 114902]
class WTMD:
    def __init__(self, nx, lx, nbar,
            eigenvalues, real_fields_idx,
            l=4,
            kc=6.02,
            dT=5.0,
            sigma_psi=0.16,
            psi_min=0.0,
            psi_max=10.0,
            dpsi=2e-3,
            update_freq=1000,
            recording_period=10000,
            u=None, up=None, I0=None, I1=None):
        self.l=l
        self.kc=kc
        self.sigma_psi=sigma_psi
        self.dT=dT
        self.psi_min=psi_min
        self.psi_max=psi_max
        self.dpsi=dpsi
        self.bins= np.round((psi_max-psi_min)/dpsi).astype(np.int)
        self.update_freq=update_freq
        self.recording_period=recording_period

        self.nx = nx
        self.lx = lx
        self.M = nx[0]*nx[1]*nx[2]
        self.Mk = nx[0]*nx[1]*(nx[2]//2+1)
        
        self.V = lx[0]*lx[1]*lx[2]
        self.CV = np.sqrt(nbar)*self.V
        
        # Choose one index of w_aux to use order parameter
        self.eigenvalues = eigenvalues
        self.real_fields_idx = real_fields_idx
        eigen_value_min = 0.0
        for count, i in enumerate(self.real_fields_idx):
            if(self.eigenvalues[i] < eigen_value_min):
                eigen_value_min = self.eigenvalues[i] 
                self.exchange_idx = i
                self.langevin_idx = count

        self.u = np.zeros(self.bins)
        self.up = np.zeros(self.bins)
        self.I0 = np.zeros(self.bins)
        self.I1 = {}
        
        # Copy data and normalize them by sqrt(nbar)*V
        if u is not None:
            self.u = u.copy()/self.CV
        if up is not None:
            self.up = up.copy()/self.CV
        if I0 is not None:
            self.I0 = I0.copy()
        if I1 is not None:
            self.I1 = I1.copy()
            for key in self.I1:
                self.I1[key] /= self.CV
        
        self.psi_range      = np.linspace(self.psi_min,             self.psi_max,             num=self.bins,   endpoint=False)
        self.psi_range_hist = np.linspace(self.psi_min-self.dpsi/2, self.psi_max-self.dpsi/2, num=self.bins+1, endpoint=True)

        # Store order parameters for updating U and U_hat 
        self.order_parameter_history = []
        # Store dH for updating I1
        self.dH_history = {}

        # Initialize arrays in Fourier spaces
        self.wt = 2*np.ones([nx[0],nx[1],nx[2]//2+1])        
        self.wt[:,:,[0, nx[2]//2]] = 1.0

        space_kx, space_ky, space_kz = np.meshgrid(
            2*np.pi/lx[0]*np.concatenate([np.arange((nx[0]+1)//2), nx[0]//2-np.arange(nx[0]//2)]),
            2*np.pi/lx[1]*np.concatenate([np.arange((nx[1]+1)//2), nx[1]//2-np.arange(nx[1]//2)]),
            2*np.pi/lx[2]*np.arange(nx[2]//2+1), indexing='ij')
        mag_k = np.sqrt(space_kx**2 + space_ky**2 + space_kz**2)
        self.fk = 1.0/(1.0 + np.exp(12.0*(mag_k-kc)/kc))
 
        # Compute fourier transform of gaussian functions
        X = self.dpsi*np.concatenate([np.arange((self.bins+1)//2), np.arange(self.bins//2)-self.bins//2])/self.sigma_psi
        self.u_kernel = np.fft.rfft(np.exp(-0.5*X**2))
        self.up_kernel = np.fft.rfft(-X/self.sigma_psi*np.exp(-0.5*X**2))
 
    # Compute order parameter Ψ
    def compute_order_parameter(self, langevin_step, w_aux):
        self.w_aux_k = np.fft.rfftn(np.reshape(w_aux[self.exchange_idx], self.nx))
        psi = np.sum(np.power(np.absolute(self.w_aux_k), self.l)*self.fk*self.wt)
        psi = np.power(psi, 1.0/self.l)/self.M
        return psi
    
    def store_order_parameter(self, psi, dH):
        self.order_parameter_history.append(psi)
        for key in dH:
            if not key in self.dH_history:
                self.dH_history[key] = []
            self.dH_history[key].append(dH[key])

    # Compute bias from psi and w_aux_k, and add it to the DH/DW
    def add_bias_to_langevin(self, psi, langevin):
        
        # Calculate current value of U'(Ψ) using linear interpolation
        up_hat = np.interp(psi, self.psi_range, self.up)
        # Calculate derivative of order parameter with respect to w_aux_k
        dpsi_dwk = np.power(np.absolute(self.w_aux_k),self.l-2.0) * np.power(psi,1.0-self.l)*self.w_aux_k*self.fk
        # Calculate derivative of order parameter with respect to w
        dpsi_dwr = np.fft.irfftn(dpsi_dwk, self.nx)*np.power(self.M, 2.0-self.l)/self.V

        # Add bias
        bias = np.reshape(self.V*up_hat*dpsi_dwr, self.M)
        langevin[self.langevin_idx] += bias
        
        print("\t[WMTD] Ψ:%8.5f, np.std(dΨ_dwr):%8.5e, np.std(bias):%8.5e: " % (psi, np.std(dpsi_dwr), np.std(bias)))

    def update_statistics(self):

        # du2 = np.zeros(self.bins)
        # dup2 = np.zeros(self.bins)
        # dI02 = np.zeros(self.bins)
        # dI12 = {}

        # # print("self.order_parameter_history", self.order_parameter_history)
        # # print("self.dH_history["A,B"]", self.dH_history["A,B"])
        # for i in range(len(self.order_parameter_history)):
        #     psi_hat = self.order_parameter_history[i]
        #     w2_hat = self.dH_history["A,B"][i]
        #     psi = self.psi_min + np.arange(self.bins)*self.dpsi
        #     TEMP = (psi_hat-psi)/self.sigma_psi
        #     amplitude = np.exp(-self.CV*self.u/self.dT)/self.CV
        #     gaussian = np.exp(-0.5*TEMP*TEMP)
        
        #     # Update u, up, I0, I1
        #     du2  += amplitude*gaussian
        #     dup2 += (TEMP/self.sigma_psi-self.CV*self.up/self.dT)*amplitude*gaussian
        #     dI02 += gaussian
        #     for key in self.dH_history:
        #         if not key in dI12:
        #             dI12[key] = np.zeros(self.bins)
        #         dI12[key] += w2_hat*gaussian/len(self.order_parameter_history)

        # du2 /= len(self.order_parameter_history)
        # dup2 /= len(self.order_parameter_history)
        # dI02 /= len(self.order_parameter_history)

        # Compute histogram
        hist, bin_edges = np.histogram(self.order_parameter_history, bins=self.psi_range_hist, density=True)
        hist_k = np.fft.rfft(hist)
        bin_mids = bin_edges[1:]-self.dpsi/2
        
        dI1 = {}
        for key in self.dH_history:
            hist_dH_, _ = np.histogram(self.order_parameter_history,
                                weights=self.dH_history[key],
                                bins=self.psi_range_hist, density=False)
            hist_dH_ /= len(self.order_parameter_history)
            hist_dH_k = np.fft.rfft(hist_dH_)
            dI1[key] = np.fft.irfft(hist_dH_k*self.u_kernel, self.bins)
        
        # Compute dU(Ψ), dU'(Ψ)
        amplitude = np.exp(-self.CV*self.u/self.dT)/self.CV
        gaussian = np.fft.irfft(hist_k*self.u_kernel, self.bins)*self.dpsi
        du  = amplitude*gaussian
        dup = amplitude*np.fft.irfft(hist_k*self.up_kernel, self.bins)*self.dpsi-self.CV*self.up/self.dT*du

        # print(np.std(du-du2))
        # print(np.std(dup-dup2))
        # print(np.std(gaussian-dI02))
        # for key in dI1:
        #     print(np.std(dI1[key]-dI12[key]))

        print("np.max(np.abs(amplitude)):%8.5e" % (np.max(np.abs(amplitude))))
        print("np.max(np.abs(gaussian)):%8.5e" % (np.max(np.abs(gaussian))))
        print("np.max(np.abs(du)):%8.5e" % (np.max(np.abs(du))))
        print("np.max(np.abs(dup)):%8.5e" % (np.max(np.abs(dup))))
        for key in dI1:
            print(f"np.max(np.abs(dI1[{key}])):%8.5e" % (np.max(np.abs(dI1[key]))))

        # Update u, up, I0, I1
        self.u  += du
        self.up += dup
        self.I0 += gaussian
        for key in dI1:
            if not key in self.I1:
                self.I1[key] = np.zeros(self.bins)
            self.I1[key] += dI1[key]

        # Reset lists
        self.order_parameter_history = []
        self.dH_history = {}
        
    def write_data(self, file_name):
        mdic = {"l":self.l,
                "kc":self.kc,
                "sigma_psi":self.sigma_psi,
                "dT":self.dT,
                "psi_min":self.psi_min,
                "psi_max":self.psi_max,
                "dpsi":self.dpsi,
                "bins":self.bins,
                "psi_range":self.psi_range,
                "update_freq":self.update_freq,
                "nx":self.nx,
                "lx":self.lx,
                "volume":self.V,
                "nbar":(self.CV/self.V)**2,
                "eigenvalues":self.eigenvalues,
                "real_fields_idx":self.real_fields_idx,
                "exchange_idx":self.exchange_idx,
                "langevin_idx":self.langevin_idx,
                "u":self.u*self.CV, "up":self.up*self.CV, "I0":self.I0}
        
        # Add I0 and dH_Ψ to the dictionary
        for key in self.I1:
            print(key)
            I1 = self.I1[key]*self.CV
            dH_psi = I1.copy()
            dH_psi[np.abs(self.I0)>0] /= self.I0[np.abs(self.I0)>0]
            monomer_pair = sorted(key.split(","))
            mdic["I1_" + monomer_pair[0] + "_" + monomer_pair[1]] = I1
            mdic["dH_psi_" + monomer_pair[0] + "_" + monomer_pair[1]] = dH_psi
        
        savemat(file_name, mdic, long_field_names=True, do_compression=True)

def calculate_sigma(langevin_nbar, langevin_dt, n_grids, volume):
        return np.sqrt(2*langevin_dt*n_grids/(volume*np.sqrt(langevin_nbar)))

class Symmetric_Polymer_Theory:
    def __init__(self, monomer_types, chi_n):
        self.monomer_types = monomer_types
        S = len(self.monomer_types)
        
        self.matrix_q = np.ones((S,S))/S
        self.matrix_p = np.identity(S) - self.matrix_q
        
        # Compute eigenvalues and orthogonal matrix
        eigenvalues, matrix_o = self.compute_eigen_system(chi_n, self.matrix_p)

        # Construct chi_n matrix
        matrix_chi = np.zeros((S,S))
        for i in range(S):
            for j in range(i+1,S):
                monomer_pair = [self.monomer_types[i], self.monomer_types[j]]
                monomer_pair.sort()
                key = monomer_pair[0] + "," + monomer_pair[1]
                if key in chi_n:
                    matrix_chi[i,j] = chi_n[key]
                    matrix_chi[j,i] = chi_n[key]

        self.matrix_chi = matrix_chi
        self.vector_s = np.matmul(matrix_chi, np.ones(S))/S
        self.vector_large_s = np.matmul(np.transpose(matrix_o), self.vector_s)

        # Indices whose auxiliary fields are real
        self.aux_fields_real_idx = []
        # Indices whose auxiliary fields are imaginary including the pressure field
        self.aux_fields_imag_idx = []
        for i in range(S-1):
            # assert(not np.isclose(eigenvalues[i], 0.0)), \
            #     "One of eigenvalues is zero for given chiN values."
            if np.isclose(eigenvalues[i], 0.0):
                print("One of eigenvalues is zero for given chiN values.")
            elif eigenvalues[i] > 0:
                self.aux_fields_imag_idx.append(i)
            else:
                self.aux_fields_real_idx.append(i)
        self.aux_fields_imag_idx.append(S-1) # add pressure field

        # The numbers of real and imaginary fields, respectively
        self.R = len(self.aux_fields_real_idx)
        self.I = len(self.aux_fields_imag_idx)

        if self.I > 1:
            print("(Warning!) For a given χN interaction parameter set, at least one of the auxiliary fields is an imaginary field. ", end="")
            print("The field fluctuations would not be fully reflected. Run this simulation at your own risk.")

        # Compute coefficients for Hamiltonian computation
        h_const, h_coef_mu1, h_coef_mu2 = self.compute_h_coef(chi_n, eigenvalues)

        # Matrix A and Inverse for converting between auxiliary fields and monomer chemical potential fields
        matrix_a = matrix_o.copy()
        matrix_a_inv = np.transpose(matrix_o).copy()/S

        # Check the inverse matrix
        error = np.std(np.matmul(matrix_a, matrix_a_inv) - np.identity(S))
        assert(np.isclose(error, 0.0)), \
            "Invalid inverse of matrix A. Perhaps matrix O is not orthogonal."

        # Compute derivatives of Hamiltonian coefficients w.r.t. χN
        epsilon = 1e-5
        self.h_const_deriv_chin = {}
        self.h_coef_mu1_deriv_chin = {}
        self.h_coef_mu2_deriv_chin = {}
        for key in chi_n:
            
            chi_n_p = chi_n.copy()
            chi_n_n = chi_n.copy()
            
            chi_n_p[key] += epsilon
            chi_n_n[key] -= epsilon

            # Compute eigenvalues and orthogonal matrix
            eigenvalues_p, _ = self.compute_eigen_system(chi_n_p, self.matrix_p)
            eigenvalues_n, _ = self.compute_eigen_system(chi_n_n, self.matrix_p)
            
            # Compute coefficients for Hamiltonian computation
            h_const_p, h_coef_mu1_p, h_coef_mu2_p = self.compute_h_coef(chi_n_p, eigenvalues_p)
            h_const_n, h_coef_mu1_n, h_coef_mu2_n = self.compute_h_coef(chi_n_n, eigenvalues_n)
            
            # Compute derivatives using finite difference
            self.h_const_deriv_chin[key] = (h_const_p - h_const_n)/(2*epsilon)
            self.h_coef_mu1_deriv_chin[key] = (h_coef_mu1_p - h_coef_mu1_n)/(2*epsilon)
            self.h_coef_mu2_deriv_chin[key] = (h_coef_mu2_p - h_coef_mu2_n)/(2*epsilon)

        self.h_const = h_const
        self.h_coef_mu1 = h_coef_mu1
        self.h_coef_mu2 = h_coef_mu2

        self.eigenvalues = eigenvalues
        self.matrix_o = matrix_o
        self.matrix_a = matrix_a
        self.matrix_a_inv = matrix_a_inv

        print("------------ Polymer Field Theory for Multimonomer ------------")
        # print("Projection matrix P:\n\t", str(self.matrix_p).replace("\n", "\n\t"))
        # print("Projection matrix Q:\n\t", str(self.matrix_q).replace("\n", "\n\t"))
        print("Eigenvalues:\n\t", self.eigenvalues)
        print("Eigenvectors [v1, v2, ...] :\n\t", str(self.matrix_o).replace("\n", "\n\t"))
        print("Mapping matrix A:\n\t", str(self.matrix_a).replace("\n", "\n\t"))
        # print("A*Inverse[A]:\n\t", str(np.matmul(self.matrix_a, self.matrix_a_inv)).replace("\n", "\n\t"))
        # print("P matrix for field residuals:\n\t", str(self.matrix_p).replace("\n", "\n\t"))

        print("Real Fields: ",      self.aux_fields_real_idx)
        print("Imaginary Fields: ", self.aux_fields_imag_idx)
        
        print("In Hamiltonian:")
        print("\treference energy: ", self.h_const)
        print("\tcoefficients of int of mu(r)/V: ", self.h_coef_mu1)
        print("\tcoefficients of int of mu(r)^2/V: ", self.h_coef_mu2)
        print("\tdH_ref/dχN: ", self.h_const_deriv_chin)
        print("\td(coef of mu(r))/dχN: ", self.h_coef_mu1_deriv_chin)
        print("\td(coef of mu(r)^2)/dχN: ", self.h_coef_mu2_deriv_chin)

    def to_aux_fields(self, w):
        return np.matmul(self.matrix_a_inv, w)

    def to_monomer_fields(self, w_aux):
        return np.matmul(self.matrix_a, w_aux)

    def compute_eigen_system(self, chi_n, matrix_p):
        S = matrix_p.shape[0]

        # Compute eigenvalues and eigenvectors
        matrix_chi = np.zeros((S,S))
        for i in range(S):
            for j in range(i+1,S):
                monomer_pair = [self.monomer_types[i], self.monomer_types[j]]
                monomer_pair.sort()
                key = monomer_pair[0] + "," + monomer_pair[1]
                if key in chi_n:
                    matrix_chi[i,j] = chi_n[key]
                    matrix_chi[j,i] = chi_n[key]
        projected_chin = np.matmul(matrix_p, np.matmul(matrix_chi, matrix_p))
        eigenvalues, eigenvectors = np.linalg.eigh(projected_chin)

        # Reordering eigenvalues and eigenvectors
        sorted_indexes = np.argsort(np.abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[sorted_indexes]
        eigenvectors = eigenvectors[:,sorted_indexes]

        # Set the last eigenvector to [1, 1, ..., 1]/√S
        eigenvectors[:,-1] = np.ones(S)/np.sqrt(S)
        
        # Make a orthogonal matrix using Gram-Schmidt
        eigen_val_0 = np.isclose(eigenvalues, 0.0, atol=1e-12)
        eigenvalues[eigen_val_0] = 0.0
        eigen_vec_0 = eigenvectors[:,eigen_val_0]
        for i in range(eigen_vec_0.shape[1]-2,-1,-1):
            vec_0 = eigen_vec_0[:,i].copy()
            for j in range(i+1, eigen_vec_0.shape[1]):
                eigen_vec_0[:,i] -= eigen_vec_0[:,j]*np.dot(vec_0,eigen_vec_0[:,j])
            eigen_vec_0[:,i] /= np.linalg.norm(eigen_vec_0[:,i])
        eigenvectors[:,eigen_val_0] = eigen_vec_0

        # Make the first element of each vector positive to restore the conventional AB polymer field theory
        for i in range(S):
            if eigenvectors[0,i] < 0.0:
                eigenvectors[:,i] *= -1.0

        # Multiply √S to eigenvectors
        eigenvectors *= np.sqrt(S)

        return eigenvalues, eigenvectors

    def compute_h_coef(self, chi_n, eigenvalues):
        S = len(self.monomer_types)

        # Compute vector X_iS
        vector_s = np.zeros(S-1)
        for i in range(S-1):
            monomer_pair = [self.monomer_types[i], self.monomer_types[S-1]]
            monomer_pair.sort()
            key = monomer_pair[0] + "," + monomer_pair[1]            
            vector_s[i] = chi_n[key]

        # Compute reference part of Hamiltonian
        h_const = 0.5*np.sum(self.vector_s)/S
        for i in range(S-1):
            if not np.isclose(eigenvalues[i], 0.0):
                h_const -= 0.5*self.vector_large_s[i]**2/eigenvalues[i]/S

        # Compute coefficients of integral of μ(r)/V
        h_coef_mu1 = np.zeros(S-1)
        for i in range(S-1):
            if not np.isclose(eigenvalues[i], 0.0):
                h_coef_mu1[i] = self.vector_large_s[i]/eigenvalues[i]

        # Compute coefficients of integral of μ(r)^2/V
        h_coef_mu2 = np.zeros(S-1)
        for i in range(S-1):
            if not np.isclose(eigenvalues[i], 0.0):
                h_coef_mu2[i] = -0.5/eigenvalues[i]*S

        return h_const, h_coef_mu1, h_coef_mu2

    # Compute total Hamiltonian
    def compute_hamiltonian(self, molecules, w_aux, total_partitions):
        S = len(self.monomer_types)

        # Compute Hamiltonian part that is related to fields
        hamiltonian_fields = -np.mean(w_aux[S-1])
        for i in range(S-1):
            hamiltonian_fields += self.h_coef_mu2[i]*np.mean(w_aux[i]**2)
            hamiltonian_fields += self.h_coef_mu1[i]*np.mean(w_aux[i])
        
        # Compute Hamiltonian part that total partition functions
        hamiltonian_partition = 0.0
        for p in range(molecules.get_n_polymer_types()):
            hamiltonian_partition -= molecules.get_polymer(p).get_volume_fraction()/ \
                            molecules.get_polymer(p).get_alpha() * \
                            np.log(total_partitions[p])

        return hamiltonian_partition + hamiltonian_fields + self.h_const

    # Compute functional derivatives of Hamiltonian w.r.t. fields of selected indices
    def compute_func_deriv(self, w_aux, phi, indices):
        S = len(self.monomer_types)
                
        elapsed_time = {}
        time_e_start = time.time()
        h_deriv = np.zeros([len(indices), w_aux.shape[1]], dtype=np.float64)
        for count, i in enumerate(indices):
            # Return dH/dw
            if i != S-1:
                h_deriv[count] += 2*self.h_coef_mu2[i]*w_aux[i]
                h_deriv[count] +=   self.h_coef_mu1[i]
                for j in range(S):
                    h_deriv[count] += self.matrix_a[j,i]*phi[self.monomer_types[j]]
            else:
                for j in range(S):
                    h_deriv[count] += phi[self.monomer_types[j]]
                h_deriv[count] -= 1.0

            # Change the sign for the imaginary fields
            if i in self.aux_fields_imag_idx:
                h_deriv[count] = -h_deriv[count]
                
        elapsed_time["h_deriv"] = time.time() - time_e_start
        
        return  h_deriv, elapsed_time

    # Compute dH/dχN
    def compute_h_deriv_chin(self, chi_n, w_aux):
        S = len(self.monomer_types)

        dH = {}
        for key in chi_n:
            dH[key] = self.h_const_deriv_chin[key]
            for i in range(S-1):
                dH[key] += self.h_coef_mu2_deriv_chin[key][i]*np.mean(w_aux[i]**2)
                dH[key] += self.h_coef_mu1_deriv_chin[key][i]*np.mean(w_aux[i])                            
        return dH

class DeepLangevinFTS:
    def __init__(self, params, random_seed=None):

        #-------------- ML Part -------------------------------- 

        # Check whether this process is the primary process of DPP in Pytorch-Lightning.
        is_secondary = os.environ.get("IS_DDP_SECONDARY")
        if is_secondary == "YES":
            self.is_secondary = True
        else:
            os.environ["IS_DDP_SECONDARY"] = "YES"
            self.is_secondary = False

        # Set the number of threads for pytorch = 1
        torch.set_num_threads(1)

        # Set torch device
        self.device_string = 'cuda:0'
        self.device = torch.device(self.device_string)

        self.training = params["training"].copy()
        self.net = None

        #-------------- Simulation Part ---------------------------------

        # Segment length
        self.monomer_types = sorted(list(params["segment_lengths"].keys()))
        self.segment_lengths = copy.deepcopy(params["segment_lengths"])
        self.distinct_polymers = copy.deepcopy(params["distinct_polymers"])
        assert(len(self.monomer_types) == len(set(self.monomer_types))), \
            "There are duplicated monomer_types"
        
        # Choose platform among [cuda, cpu-mkl]
        avail_platforms = PlatformSelector.avail_platforms()
        if "platform" in params:
            platform = params["platform"]
        elif "cpu-mkl" in avail_platforms and len(params["nx"]) == 1: # for 1D simulation, use CPU
            platform = "cpu-mkl"
        elif "cuda" in avail_platforms: # If cuda is available, use GPU
            platform = "cuda"
        else:
            platform = avail_platforms[0]

        # (C++ class) Create a factory for given platform and chain_model
        if "reduce_gpu_memory_usage" in params and platform == "cuda":
            factory = PlatformSelector.create_factory(platform, params["reduce_gpu_memory_usage"])
        else:
            factory = PlatformSelector.create_factory(platform, False)
        factory.display_info()

        # (C++ class) Computation box
        cb = factory.create_computation_box(params["nx"], params["lx"])

        # Flory-Huggins parameters, χN
        self.chi_n = {}
        for monomer_pair_str, chin_value in params["chi_n"].items():
            monomer_pair = re.split(',| |_|/', monomer_pair_str)
            assert(monomer_pair[0] in self.segment_lengths), \
                f"Monomer type '{monomer_pair[0]}' is not in 'segment_lengths'."
            assert(monomer_pair[1] in self.segment_lengths), \
                f"Monomer type '{monomer_pair[1]}' is not in 'segment_lengths'."
            assert(monomer_pair[0] != monomer_pair[1]), \
                "Do not add self interaction parameter, " + monomer_pair_str + "."
            monomer_pair.sort()
            sorted_monomer_pair = monomer_pair[0] + "," + monomer_pair[1] 
            assert(not sorted_monomer_pair in self.chi_n), \
                f"There are duplicated χN ({sorted_monomer_pair}) parameters."
            self.chi_n[sorted_monomer_pair] = chin_value

        for monomer_pair in itertools.combinations(self.monomer_types, 2):
            monomer_pair = list(monomer_pair)
            monomer_pair.sort()
            sorted_monomer_pair = monomer_pair[0] + "," + monomer_pair[1] 
            if not sorted_monomer_pair in self.chi_n:
                self.chi_n[sorted_monomer_pair] = 0.0
                
        # Multimonomer polymer field theory
        self.mpt = Symmetric_Polymer_Theory(self.monomer_types, self.chi_n)
        
        # The numbers of real and imaginary fields, respectively
        self.R = len(self.mpt.aux_fields_real_idx)
        self.I = len(self.mpt.aux_fields_imag_idx)

        # Initialize well-tempered metadynamics class
        if "wtmd" in params:
            self.wtmd = WTMD(
                nx   = params["nx"],
                lx   = params["lx"],
                nbar = params["langevin"]["nbar"],
                eigenvalues      = self.mpt.eigenvalues,
                real_fields_idx  = self.mpt.aux_fields_real_idx,
                l                = params["wtmd"]["l"],
                kc               = params["wtmd"]["kc"],
                dT               = params["wtmd"]["dT"],
                sigma_psi        = params["wtmd"]["sigma_psi"],
                psi_min          = params["wtmd"]["psi_min"],
                psi_max          = params["wtmd"]["psi_max"],
                dpsi             = params["wtmd"]["dpsi"],
                update_freq      = params["wtmd"]["update_freq"],
                recording_period = params["wtmd"]["recording_period"],
            )
        else:
            self.wtmd = None

        # Total volume fraction
        assert(len(self.distinct_polymers) >= 1), \
            "There is no polymer chain."

        total_volume_fraction = 0.0
        for polymer in self.distinct_polymers:
            total_volume_fraction += polymer["volume_fraction"]
        assert(np.isclose(total_volume_fraction,1.0)), "The sum of volume fractions must be equal to 1."

        # Polymer chains
        for polymer_counter, polymer in enumerate(self.distinct_polymers):
            blocks_input = []
            alpha = 0.0             # total_relative_contour_length
            has_node_number = not "v" in polymer["blocks"][0]
            for block in polymer["blocks"]:
                alpha += block["length"]
                if has_node_number:
                    assert(not "v" in block), \
                        "Index v should exist in all blocks, or it should not exist in all blocks for each polymer." 
                    assert(not "u" in block), \
                        "Index u should exist in all blocks, or it should not exist in all blocks for each polymer." 
                    blocks_input.append([block["type"], block["length"], len(blocks_input), len(blocks_input)+1])
                else:
                    assert("v" in block), \
                        "Index v should exist in all blocks, or it should not exist in all blocks for each polymer." 
                    assert("u" in block), \
                        "Index u should exist in all blocks, or it should not exist in all blocks for each polymer." 
                    blocks_input.append([block["type"], block["length"], block["v"], block["u"]])
            polymer.update({"blocks_input":blocks_input})

        # Random copolymer chains
        self.random_fraction = {}
        for polymer in self.distinct_polymers:

            is_random = False
            for block in polymer["blocks"]:
                if "fraction" in block:
                    is_random = True
            if not is_random:
                continue

            assert(len(polymer["blocks"]) == 1), \
                "Only single block random copolymer is allowed."

            statistical_segment_length = 0
            total_random_fraction = 0
            for monomer_type in polymer["blocks"][0]["fraction"]:
                statistical_segment_length += self.segment_lengths[monomer_type]**2 * polymer["blocks"][0]["fraction"][monomer_type]
                total_random_fraction += polymer["blocks"][0]["fraction"][monomer_type]
            statistical_segment_length = np.sqrt(statistical_segment_length)

            assert(np.isclose(total_random_fraction, 1.0)), \
                "The sum of volume fractions of random copolymer must be equal to 1."

            random_type_string = polymer["blocks"][0]["type"]
            assert(not random_type_string in self.segment_lengths), \
                f"The name of random copolymer '{random_type_string}' is already used as a type in 'segment_lengths' or other random copolymer"

            # Add random copolymers
            polymer["block_monomer_types"] = [random_type_string]
            self.segment_lengths.update({random_type_string:statistical_segment_length})
            self.random_fraction[random_type_string] = polymer["blocks"][0]["fraction"]

        # Make a monomer color dictionary
        dict_color= {}
        colors = ["red", "blue", "green", "cyan", "magenta", "yellow"]
        for count, type in enumerate(self.segment_lengths.keys()):
            if count < len(colors):
                dict_color[type] = colors[count]
            else:
                dict_color[type] = np.random.rand(3,)
        print("Monomer color: ", dict_color)
            
        # Draw polymer chain architectures
        for idx, polymer in enumerate(self.distinct_polymers):
        
            # Make a graph
            G = nx.Graph()
            for block in polymer["blocks_input"]:
                type = block[0]
                length = round(block[1]/params["ds"])
                v = block[2]
                u = block[3]
                G.add_edge(v, u, weight=length, monomer_type=type)

            # Set node colors
            color_map = []
            for node in G:
                if len(G.edges(node)) == 1:
                    color_map.append('yellow')
                else: 
                    color_map.append('gray')

            labels = nx.get_edge_attributes(G, 'weight')
            pos = nx.drawing.nx_agraph.graphviz_layout(G, prog='twopi')
            colors = [dict_color[G[u][v]['monomer_type']] for u,v in G.edges()]

            plt.figure(figsize=(20,20))
            title = "Polymer ID: %2d," % (idx)
            title += "\nColors of monomers: " + str(dict_color) + ","
            title += "\nColor of chain ends: 'yellow',"
            title += "\nColor of junctions: 'gray',"
            title += "\nPlease note that the length of each edge is not proportional to the number of monomers in this image."
            plt.title(title)
            nx.draw(G, pos, node_color=color_map, edge_color=colors, width=4, with_labels=True) #, node_size=100, font_size=15)
            nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, rotate=False, bbox=dict(boxstyle='round', ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0), alpha=0.5)) #, font_size=12)
            plt.savefig("polymer_%01d.png" % (idx))

        # (C++ class) Molecules list
        molecules = factory.create_molecules_information(params["chain_model"], params["ds"], self.segment_lengths)

        # Add polymer chains
        for polymer in self.distinct_polymers:
            molecules.add_polymer(polymer["volume_fraction"], polymer["blocks_input"])

        # (C++ class) Propagator Analyzer
        if "aggregate_propagator_computation" in params:
            propagator_analyzer = factory.create_propagator_analyzer(molecules, params["aggregate_propagator_computation"])
        else:
            propagator_analyzer = factory.create_propagator_analyzer(molecules, True)

        # Standard deviation of normal noise of Langevin dynamics
        langevin_sigma = calculate_sigma(params["langevin"]["nbar"], params["langevin"]["dt"], np.prod(params["nx"]), np.prod(params["lx"]))

        # dH/dw_aux[i] is scaled by dt_scaling[i]
        S = len(self.monomer_types)
        self.dt_scaling = np.ones(S)
        for i in range(S-1):
            self.dt_scaling[i] = np.abs(self.mpt.eigenvalues[i])/np.max(np.abs(self.mpt.eigenvalues))

        # Set random generator
        if random_seed is None:         
            self.random_bg = np.random.PCG64()  # Set random bit generator
        else:
            self.random_bg = np.random.PCG64(random_seed)
        self.random = np.random.Generator(self.random_bg)

        # -------------- print simulation parameters ------------
        print("---------- Simulation Parameters ----------")
        print("Platform :", "cuda")
        print("Box Dimension: %d" % (cb.get_dim()))
        print("Nx:", cb.get_nx())
        print("Lx:", cb.get_lx())
        print("dx:", cb.get_dx())
        print("Volume: %f" % (cb.get_volume()))

        print("Chain model: %s" % (params["chain_model"]))
        print("Segment lengths:\n\t", list(self.segment_lengths.items()))
        print("Conformational asymmetry (epsilon): ")
        for monomer_pair in itertools.combinations(self.monomer_types,2):
            print("\t%s/%s: %f" % (monomer_pair[0], monomer_pair[1], self.segment_lengths[monomer_pair[0]]/self.segment_lengths[monomer_pair[1]]))

        print("χN: ")
        for key in self.chi_n:
            print("\t%s: %f" % (key, self.chi_n[key]))

        for p in range(molecules.get_n_polymer_types()):
            print("distinct_polymers[%d]:" % (p) )
            print("\tvolume fraction: %f, alpha: %f, N: %d" %
                (molecules.get_polymer(p).get_volume_fraction(),
                 molecules.get_polymer(p).get_alpha(),
                 molecules.get_polymer(p).get_n_segment_total()))

        print("Invariant Polymerization Index (N_Ref): %d" % (params["langevin"]["nbar"]))
        print("Langevin Sigma: %f" % (langevin_sigma))
        print("Scaling factor of delta tau N for each field: ", self.dt_scaling)
        print("Random Number Generator: ", self.random_bg.state)

        propagator_analyzer.display_blocks()
        propagator_analyzer.display_propagators()

        #  Save Internal Variables
        self.params = params
        self.chain_model = params["chain_model"]
        self.ds = params["ds"]
        self.langevin = params["langevin"].copy()
        self.langevin.update({"sigma":langevin_sigma})
        self.langevin.pop("max_step", None)

        self.verbose_level = params["verbose_level"]
        self.saddle = params["saddle"].copy()
        self.recording = params["recording"].copy()

        self.factory = factory
        self.cb = cb
        self.molecules = molecules
        self.propagator_analyzer = propagator_analyzer
        self.solver = None
        self.am = None
        
    def create_solvers(self):
        
        params = self.params
        
        # (C++ class) Solver using Pseudo-spectral method
        self.solver = self.factory.create_pseudospectral_solver(self.cb, self.molecules, self.propagator_analyzer)

        # (C++ class) Fields relaxation using Anderson Mixing
        self.am = self.factory.create_anderson_mixing(
            len(self.mpt.aux_fields_imag_idx)*np.prod(params["nx"]),   # the number of variables
            params["am"]["max_hist"],                                   # maximum number of history
            params["am"]["start_error"],                                # when switch to AM from simple mixing
            params["am"]["mix_min"],                                    # minimum mixing rate of simple mixing
            params["am"]["mix_init"])                                   # initial mixing rate of simple mixing

    def compute_concentrations(self, w_aux):
        S = len(self.monomer_types)
        elapsed_time = {}

        # Convert auxiliary fields to monomer fields
        w = self.mpt.to_monomer_fields(w_aux)

        # Make a dictionary for input fields 
        w_input = {}
        for i in range(S):
            w_input[self.monomer_types[i]] = w[i]
        for random_polymer_name, random_fraction in self.random_fraction.items():
            w_input[random_polymer_name] = np.zeros(self.cb.get_n_grid(), dtype=np.float64)
            for monomer_type, fraction in random_fraction.items():
                w_input[random_polymer_name] += w_input[monomer_type]*fraction

        # For the given fields, compute propagators
        time_solver_start = time.time()
        self.solver.compute_propagators(w_input)
        elapsed_time["solver"] = time.time() - time_solver_start

        # Compute concentrations for each monomer type
        time_phi_start = time.time()
        phi = {}
        self.solver.compute_concentrations()
        for monomer_type in self.monomer_types:
            phi[monomer_type] = self.solver.get_total_concentration(monomer_type)

        # Add random copolymer concentration to each monomer type
        for random_polymer_name, random_fraction in self.random_fraction.items():
            phi[random_polymer_name] = self.solver.get_total_concentration(random_polymer_name)
            for monomer_type, fraction in random_fraction.items():
                phi[monomer_type] += phi[random_polymer_name]*fraction
        elapsed_time["phi"] = time.time() - time_phi_start

        return phi, elapsed_time

    def save_training_data(self, path, w_real, h_deriv, w_diff):

        # Make a dictionary for chi_n
        chi_n_mat = {}
        for key in self.chi_n:
            chi_n_mat[key] = self.chi_n[key]

        np.savez_compressed(path,
            dim=self.cb.get_dim(), nx=self.cb.get_nx(), lx=self.cb.get_lx(),
            chi_n=chi_n_mat, chain_model=self.chain_model, ds=self.ds,
            dt=self.langevin["dt"], nbar=self.langevin["nbar"], params=self.params,
            w_real=w_real.astype(np.float16), w_real_std=np.std(w_real, axis=1),
            h_deriv=h_deriv.astype(np.float16), h_deriv_std=np.std(h_deriv, axis=1),
            w_diff=w_diff.astype(np.float16), w_diff_std=np.std(w_diff, axis=1))

    def save_simulation_data(self, path, w, phi, langevin_step, normal_noise_prev):
        
        # Make a dictionary for chi_n
        chi_n_mat = {}
        for key in self.chi_n:
            chi_n_mat[key] = self.chi_n[key]

        # Make dictionary for data
        mdic = {"dim":self.cb.get_dim(), "nx":self.cb.get_nx(), "lx":self.cb.get_lx(),
            "chi_n":chi_n_mat, "chain_model":self.chain_model, "ds":self.ds,
            "dt":self.langevin["dt"], "nbar":self.langevin["nbar"], "initial_params": self.params,
            "eigenvalues": self.mpt.eigenvalues,
            "aux_fields_real": self.mpt.aux_fields_real_idx,
            "aux_fields_imag": self.mpt.aux_fields_imag_idx,
            "matrix_a": self.mpt.matrix_a, "matrix_a_inverse": self.mpt.matrix_a_inv, 
            "langevin_step":langevin_step,
            "random_generator":self.random_bg.state["bit_generator"],
            "random_state_state":str(self.random_bg.state["state"]["state"]),
            "random_state_inc":str(self.random_bg.state["state"]["inc"]),
            "normal_noise_prev":normal_noise_prev, "monomer_types":self.monomer_types}
        
        # Add w fields to the dictionary
        for i, name in enumerate(self.monomer_types):
            mdic["w_" + name] = w[i]
        
        # Add concentrations to the dictionary
        for name in self.monomer_types:
            mdic["phi_" + name] = phi[name]
        
        # Save data with matlab format
        savemat(path, mdic, long_field_names=True, do_compression=True)

    def make_training_dataset(self, initial_fields, final_fields_configuration_file_name):

        # Skip if this process is not the primary process of DPP in Pytorch-Lightning.
        # This is because DDP duplicates the main script when using multiple GPUs.
        if self.is_secondary:
            return

        # Create solver and anderson mixing solvers if necessary
        if self.solver is None:
            self.create_solvers()

        print("---------- Make Training Dataset ----------")

        # The number of components
        S = len(self.monomer_types)

        # The numbers of real and imaginary fields respectively
        R = self.R
        I = self.I
        
        # Training data directory
        pathlib.Path(self.training["data_dir"]).mkdir(parents=True, exist_ok=True)

        # Reshape initial fields
        w = np.zeros([S, self.cb.get_n_grid()], dtype=np.float64)
        for i in range(S):
            w[i] = np.reshape(initial_fields[self.monomer_types[i]],  self.cb.get_n_grid())

        # Convert monomer chemical potential fields into auxiliary fields
        w_aux = self.mpt.to_aux_fields(w)

        # Find saddle point 
        print("iterations, mass error, total partitions, total energy, incompressibility error (or saddle point error)")
        _, phi, _, _, _, _, _ = self.find_saddle_point(w_aux=w_aux, tolerance=self.saddle["tolerance"], net=None)

        # Create an empty array for field update algorithm
        normal_noise_prev = np.zeros([R, self.cb.get_n_grid()], dtype=np.float64)

        # The number of times that 'find_saddle_point' has failed to find a saddle point
        saddle_fail_count = 0
        successive_fail_count = 0
        tol_training_fail_count = 0

        # Langevin iteration begins here
        for langevin_step in range(1, self.training["max_step"]+1):
            print("Langevin step: ", langevin_step)

            # Copy data for restoring
            w_aux_copy = w_aux.copy()
            phi_copy = phi.copy()

            # Compute functional derivatives of Hamiltonian w.r.t. real-valued fields 
            w_lambda, _ = self.mpt.compute_func_deriv(w_aux, phi, self.mpt.aux_fields_real_idx)

            # Update w_aux using Leimkuhler-Matthews method
            normal_noise_current = self.random.normal(0.0, self.langevin["sigma"], [R, self.cb.get_n_grid()])
            for count, i in enumerate(self.mpt.aux_fields_real_idx):
                scaling = self.dt_scaling[i]
                w_aux[i] += -w_lambda[count]*self.langevin["dt"]*scaling + 0.5*(normal_noise_prev[count] + normal_noise_current[count])*np.sqrt(scaling)

            # Swap two noise arrays
            normal_noise_prev, normal_noise_current = normal_noise_current, normal_noise_prev
            
            # Find saddle point
            w_imag_start = w_aux[self.mpt.aux_fields_imag_idx].copy()
            w_imag, phi, _, _, _, _, _ = self.find_saddle_point(w_aux=w_aux, tolerance=self.saddle["tolerance"], net=None)

            # Compute functional derivatives of Hamiltonian w.r.t. imaginary fields 
            h_deriv, _ = self.mpt.compute_func_deriv(w_aux, phi, self.mpt.aux_fields_imag_idx)

            # Compute total error
            error_level_array = np.std(h_deriv, axis=1)
            error_level = np.max(error_level_array)

            # If the tolerance of the saddle point was not met, regenerate Langevin random noise and continue
            if np.isnan(error_level) or error_level >= self.saddle["tolerance"]:
                if successive_fail_count < 5:                
                    print("The tolerance of the saddle point was not met. Langevin random noise is regenerated.")

                    # Restore w_aux and phi
                    w_aux = w_aux_copy
                    phi = phi_copy
                    
                    # Increment counts and continue
                    successive_fail_count += 1
                    saddle_fail_count += 1
                    continue
                else:
                    sys.exit("The tolerance of the saddle point was not met %d times in a row. Simulation is aborted." % (successive_fail_count))
            else:
                successive_fail_count = 0

            # Training data is sampled from random noise distribution with various standard deviations
            if (langevin_step % self.training["recording_period"] == 0):
                w_imag_tol = w_imag.copy()
                
                # Find a more accurate saddle point
                w_imag, phi_ref, _, _, _, _, _ = self.find_saddle_point(w_aux=w_aux, tolerance=self.training["tolerance"], net=None)
                w_imag_ref = w_imag.copy()
                
                # Compute functional derivative of H w.r.t. imaginary part of w_aux
                h_deriv, _ = self.mpt.compute_func_deriv(w_aux, phi_ref, self.mpt.aux_fields_imag_idx)

                # compute error level
                error_level_array = np.std(h_deriv, axis=1)
                error_level = np.max(error_level_array)

                # If Anderson mixing fails to find a more accurate saddle point, continue
                if np.isnan(error_level) or error_level > self.training["tolerance"]:
                    print("Anderson mixing has failed to find a more accurate saddle point. Training data generation is skipped at this Langevin step.")
                    
                    # Restore w_aux using w_imag_tol
                    w_aux[self.mpt.aux_fields_imag_idx] = w_imag_tol
                    
                    tol_training_fail_count += 1
                    continue
                
                # Differences of imaginary fields from accurate values
                w_imag_diff_start = w_imag_start - w_imag_ref
                w_imag_diff_tol = w_imag_tol - w_imag_ref

                # Range of sigma of random noise
                sigma_max = np.std(w_imag_diff_start, axis=1)
                sigma_min = np.std(w_imag_diff_tol, axis=1)
                
                print("sigma_max: ", sigma_max)
                print("sigma_min: ", sigma_min)
                
                # Select noise levels between sigma a and sigma b for each imaginary field
                sigma_array = np.zeros([I, self.training["recording_n_data"]-1])
                for i in range(I):
                    log_sigma_sample = np.random.uniform(np.log(sigma_min[i]), np.log(sigma_max[i]), self.training["recording_n_data"]-1)
                    sigma_array[i] = np.exp(log_sigma_sample)

                # print(sigma_array)
                #print(np.log(sigma_list))
                print("File name, Standard deviations of the generated noise: ")
                for std_idx in range(self.training["recording_n_data"]):
                    path = os.path.join(self.training["data_dir"], "%05d_%03d.npz" % (langevin_step, std_idx))
                    
                    # Copy w_imag_diff_start
                    if std_idx == 0:
                        print(path)
                        noise = w_imag_diff_start.copy()
                    
                    # Generate random noise
                    else:
                        sigma = sigma_array[:,std_idx-1]
                        print(path, end=", ")
                        print(sigma)
                        noise = np.zeros([I, self.cb.get_n_grid()])
                        for i in range(I):
                            noise[i] = np.random.normal(0, sigma[i], self.cb.get_n_grid())
                    
                    # Add noise
                    w_imag_with_noise = w_imag_ref.copy()
                    for i in range(I):
                        w_imag_with_noise[i] += noise[i]

                    # Add random noise and convert to monomer chemical potential fields
                    w_aux_noise = w_aux.copy()
                    w_aux_noise[self.mpt.aux_fields_imag_idx] = w_imag_with_noise
                    
                    # Compute total concentrations with noised w_aux
                    phi_noise, _ = self.compute_concentrations(w_aux_noise)

                    # Compute functional derivative of H w.r.t. imaginary fields 
                    h_deriv, _ = self.mpt.compute_func_deriv(w_aux_noise, phi_noise, self.mpt.aux_fields_imag_idx)

                    # for i in range(I):
                    #     print("%d: %7.2e, %7.2e" % (i, np.std(noise[i]), np.std(h_deriv[i])))
                    self.save_training_data(path, w_aux_noise[self.mpt.aux_fields_real_idx], h_deriv, -noise)
            
            # Save training check point
            if (langevin_step) % self.recording["recording_period"] == 0:
                w = self.mpt.to_monomer_fields(w_aux)
                self.save_simulation_data(
                    path=os.path.join(self.training["data_dir"], "fields_%06d.mat" % (langevin_step)),
                    w=w, phi=phi, langevin_step=langevin_step,
                    normal_noise_prev=normal_noise_prev)

        print( "The number of times that Anderson mixing could not find a saddle point and generated Langevin random noise: %d times" % 
            (saddle_fail_count))

        print( "The number of times that Anderson mixing could not find a more accurate saddle point and skipped training data generation: %d times" % 
            (tol_training_fail_count))

        # Save final configuration to use it as input in actual simulation
        w = self.mpt.to_monomer_fields(w_aux)
        self.save_simulation_data(path=final_fields_configuration_file_name, 
            w=w, phi=phi, langevin_step=0, 
            normal_noise_prev=normal_noise_prev)

    def train_model(self, model_file=None, epoch_offset=None):

        # Free allocated gpu memory for solver and anderson mixing
        if type(self.solver) != type(None):
            self.solver = None
            self.am = None

        torch.set_num_threads(1)

        print("---------- Training Parameters ----------")
        data_dir = self.training["data_dir"]
        batch_size = self.training["batch_size"]
        num_workers = self.training["num_workers"]
        gpus = self.training["gpus"] 
        num_nodes = self.training["num_nodes"] 
        max_epochs = self.training["max_epochs"] 
        lr = self.training["lr"]
        precision = self.training["precision"]
        features = self.training["features"]

        print(f"data_dir: {data_dir}, batch_size: {batch_size}, num_workers: {num_workers}")
        print(f"gpus: {gpus}, num_nodes: {num_nodes}, max_epochs: {max_epochs}, precision: {precision}")
        print(f"learning_rate: {lr}")

        print(f"---------- Model File : {model_file} ----------")
        assert((model_file is None and epoch_offset is None) or
             (model_file is not None and epoch_offset is not None)), \
            "To continue the training, put both model file name and epoch offset."

        # The number of input channels
        in_channels = self.R+2*self.I

        # Training data
        train_dataset = FtsDataset(data_dir, self.R, self.I)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
        print(len(train_dataset))

        # Create NN
        if model_file:                                                    
            self.net = TrainAndInference(dim=self.cb.get_dim(), in_channels=in_channels, mid_channels=features, out_channels=self.I, lr=lr, epoch_offset=epoch_offset+1)
            params_weight, normalization_factor = torch.load(model_file, map_location=self.device_string)
            self.net.load_state_dict(params_weight, strict=True)
            self.net.set_normalization_factor(normalization_factor)
            max_epochs -= epoch_offset+1
        else:
            self.net = TrainAndInference(dim=self.cb.get_dim(), in_channels=in_channels, mid_channels=features, out_channels=self.I, lr=lr)
            normalization_factor = train_dataset.get_normalization_factor()
            self.net.set_normalization_factor(normalization_factor)
                    
        # Training NN
        trainer = pl.Trainer(accelerator="gpu", devices=gpus,
                num_nodes=num_nodes, max_epochs=max_epochs, precision=precision,
                strategy=DDPStrategy(process_group_backend="gloo", find_unused_parameters=False),
                # process_group_backend="nccl" or "gloo"
                benchmark=True, log_every_n_steps=5)
        trainer.fit(self.net, train_loader, None)

        # Terminate all secondary processes of DDP
        if self.is_secondary:
            exit()

    def find_best_epoch(self, initial_fields, best_epoch_file_name):

        # Skip if this process is not the primary process of DPP in Pytorch-Lightning.
        # This is because DDP duplicates the main script when using multiple GPUs.
        if self.is_secondary:
            return
        
        print("---------- Finding the Best Epoch ----------")

        # -------------- deep learning --------------
        saved_weight_dir = self.training["model_dir"]
        torch.set_num_threads(1)

        #-------------- test roughly ------------
        list_saddle_iter_per = []
        file_list = sorted(glob.glob(saved_weight_dir + "/*.pth"), key=lambda l: (len(l), l))
        print(file_list)
        print("iteration, mass error, total_partition, hamiltonian, error_level")
        for model_file in file_list:
            saddle_iter_per, total_error_iter_per = self.run(initial_fields=initial_fields, max_step=10, model_file=model_file)
            if not np.isnan(total_error_iter_per):
                list_saddle_iter_per.append([model_file, saddle_iter_per])
        sorted_saddle_iter_per = sorted(list_saddle_iter_per, key=lambda l:l[1])

        #-------------- test top-10 epochs ------------
        list_saddle_iter_per = []
        for data in sorted_saddle_iter_per[0:10]:
            model_file = data[0]
            saddle_iter_per, total_error_iter_per = self.run(initial_fields=initial_fields, max_step=100, model_file=model_file)
            if not np.isnan(total_error_iter_per):
                list_saddle_iter_per.append([model_file, saddle_iter_per, total_error_iter_per])

        sorted_saddle_iter_per = sorted(list_saddle_iter_per, key=lambda l:(l[1], l[2]))
        print("\n\tfile name:    # iterations per langevin step,    total error per langevin step")
        for saddle_iter in sorted_saddle_iter_per:
            print("'%s': %5.2f, %12.3E" % tuple(saddle_iter), end = "\n")
        shutil.copy2(sorted_saddle_iter_per[0][0], best_epoch_file_name)
        print(f"\n'{sorted_saddle_iter_per[0][0]}' has been copied as '{best_epoch_file_name}'")

    def continue_run(self, field_file_name, max_step, final_fields_configuration_file_name="LastLangevinStep.mat", prefix="", model_file=None, use_wtmd=False, wtmd_file_name=None):

        # Skip if this process is not the primary process of DPP in Pytorch-Lightning.
        # This is because DDP duplicates the main script when using multiple GPUs.
        if self.is_secondary:
            return

        # The number of components
        S = len(self.monomer_types)

        # Load_data
        field_data = loadmat(field_file_name, squeeze_me=True)
        
        # Check if field_data["langevin_step"] is a multiple of self.recording["sf_recording_period"]
        if field_data["langevin_step"] % self.recording["sf_recording_period"] != 0:
            print(f"(Warning!) 'langevin_step' of {field_file_name} is not a multiple of 'sf_recording_period'.")
            next_sf_langevin_step = (field_data["langevin_step"]//self.recording["sf_recording_period"] + 1)*self.recording["sf_recording_period"]
            print(f"The structure function will be correctly recorded after {next_sf_langevin_step}th langevin_step." )

        # Restore random state
        self.random_bg.state ={'bit_generator': 'PCG64',
            'state': {'state': int(field_data["random_state_state"]),
                      'inc':   int(field_data["random_state_inc"])},
                      'has_uint32': 0, 'uinteger': 0}
        print("Restored Random Number Generator: ", self.random_bg.state)

        # Make initial_fields
        initial_fields = {}
        for name in self.monomer_types:
            initial_fields[name] = np.array(field_data["w_" + name])

        # Restore WTMD statistics
        if use_wtmd:
            wtmd_data = loadmat(wtmd_file_name, squeeze_me=True)

            I1 = {}
            for key in self.chi_n:
                monomer_pair = sorted(key.split(","))
                I1[key] =wtmd_data["I1_" + monomer_pair[0] + "_" + monomer_pair[1]]
            
            self.wtmd = WTMD(
                nx   = self.cb.get_nx(),
                lx   = self.cb.get_lx(),
                nbar = self.langevin["nbar"],
                eigenvalues      = self.mpt.eigenvalues,
                real_fields_idx  = self.mpt.aux_fields_real_idx,
                l                = wtmd_data["l"],
                kc               = wtmd_data["kc"],
                dT               = wtmd_data["dT"],
                sigma_psi        = wtmd_data["sigma_psi"],
                psi_min          = wtmd_data["psi_min"],
                psi_max          = wtmd_data["psi_max"],
                dpsi             = wtmd_data["dpsi"],
                update_freq      = wtmd_data["update_freq"],
                recording_period = self.params["wtmd"]["recording_period"],
                u = wtmd_data["u"],
                up = wtmd_data["up"],
                I0 = wtmd_data["I0"],
                I1 = I1,
            )

        # Run
        self.run(initial_fields=initial_fields,
            max_step=max_step, model_file=model_file,
            prefix=prefix, final_fields_configuration_file_name=final_fields_configuration_file_name,
            use_wtmd = use_wtmd,
            normal_noise_prev=field_data["normal_noise_prev"],
            start_langevin_step=field_data["langevin_step"]+1)

    def run(self, initial_fields, max_step, final_fields_configuration_file_name="LastLangevinStep.mat", prefix="", use_wtmd=False, model_file=None, normal_noise_prev=None, start_langevin_step=None):

        # ------------ ML Part ------------------
        # Skip if this process is not the primary process of DPP in Pytorch-Lightning.
        # This is because DDP duplicates the main script when using multiple GPUs.
        if self.is_secondary:
            return

        # The number of input channels 
        in_channels = self.R+2*self.I

        # Load deep learning model weights
        print(f"---------- Model File : {model_file} ----------")
        if model_file :
            self.net = TrainAndInference(dim=self.cb.get_dim(), in_channels=in_channels, mid_channels=self.training["features"], out_channels=self.I)
            params_weight, normalization_factor = torch.load(model_file, map_location =self.device_string)
            self.net.load_state_dict(params_weight, strict=True)
            self.net.set_normalization_factor(normalization_factor)
            self.net.set_inference_mode(self.device)

        # ------------ Simulation Part ------------------
        # Create solver and anderson mixing solvers if necessary
        if self.solver is None:
            self.create_solvers()

        print("---------- Run  ----------")

        # The number of components
        S = len(self.monomer_types)

        # The numbers of real and imaginary fields, respectively
        R = self.R
        I = self.I

        # Simulation data directory
        pathlib.Path(self.recording["dir"]).mkdir(parents=True, exist_ok=True)

        # Reshape initial fields
        w = np.zeros([S, self.cb.get_n_grid()], dtype=np.float64)
        for i in range(S):
            w[i] = np.reshape(initial_fields[self.monomer_types[i]],  self.cb.get_n_grid())
            
        # Convert monomer chemical potential fields into auxiliary fields
        w_aux = self.mpt.to_aux_fields(w)

        # Find saddle point 
        print("iterations, mass error, total partitions, Hamiltonian, incompressibility error (or saddle point error)")
        _, phi, _, _, _, _, _ = self.find_saddle_point(w_aux=w_aux, tolerance=self.saddle["tolerance"], net=None)

        # Dictionary to record history of H and dH/dχN
        H_history = []
        dH_history = {}
        for key in self.chi_n:
            dH_history[key] = []

        # Arrays for structure function
        sf_average = {} # <u(k) phi(-k)>
        for monomer_id_pair in itertools.combinations_with_replacement(list(range(S)),2):
            sorted_pair = sorted(monomer_id_pair)
            type_pair = self.monomer_types[sorted_pair[0]] + "," + self.monomer_types[sorted_pair[1]]
            sf_average[type_pair] = np.zeros_like(np.fft.rfftn(np.reshape(w[0], self.cb.get_nx())), np.complex128)

        # Create an empty array for field update algorithm
        if normal_noise_prev is None :
            normal_noise_prev = np.zeros([R, self.cb.get_n_grid()], dtype=np.float64)
        else:
            normal_noise_prev = normal_noise_prev

        if start_langevin_step is None :
            start_langevin_step = 1

        total_saddle_iter = 0
        total_error_level = 0.0
        total_net_failed = 0

        # The number of times that 'find_saddle_point' has failed to find a saddle point
        saddle_fail_count = 0
        successive_fail_count = 0

        # Init timers
        total_elapsed_time = {}
        total_elapsed_time["neural_net"] = 0.0
        total_elapsed_time["solver"] = 0.0
        total_elapsed_time["phi"] = 0.0
        total_elapsed_time["am"] = 0.0
        total_elapsed_time["hamiltonian"] = 0.0
        total_elapsed_time["h_deriv"] = 0.0
        total_elapsed_time["langevin"] = 0.0

        time_start = time.time()
        # Langevin iteration begins here
        for langevin_step in range(start_langevin_step, max_step+1):
            print("Langevin step: ", langevin_step)

            # Copy data for restoring
            w_aux_copy = w_aux.copy()
            phi_copy = phi.copy()

            time_langevin = time.time()

            # Compute functional derivatives of Hamiltonian w.r.t. real-valued fields 
            w_lambda, _ = self.mpt.compute_func_deriv(w_aux, phi, self.mpt.aux_fields_real_idx)

            # # Print standard deviation of w_aux and w_lambda
            # for count, i in enumerate(self.mpt.aux_fields_real_idx):
            #     print("i, count, np.std(w_aux[i]), np.std(w_lambda[count])", i, count, np.std(w_aux[i]), np.std(w_lambda[count]))

            if use_wtmd:
                # Compute order parameter
                psi = self.wtmd.compute_order_parameter(langevin_step, w_aux)
                
                # Add bias to w_lambda
                self.wtmd.add_bias_to_langevin(psi, w_lambda)

            # Update w_aux using Leimkuhler-Matthews method
            normal_noise_current = self.random.normal(0.0, self.langevin["sigma"], [R, self.cb.get_n_grid()])
            for count, i in enumerate(self.mpt.aux_fields_real_idx):
                scaling = self.dt_scaling[i]
                w_aux[i] += -w_lambda[count]*self.langevin["dt"]*scaling + 0.5*(normal_noise_prev[count] + normal_noise_current[count])*np.sqrt(scaling)

            # Swap two noise arrays
            normal_noise_prev, normal_noise_current = normal_noise_current, normal_noise_prev

            total_elapsed_time["langevin"] += time.time() - time_langevin

            # Find saddle point of the pressure field
            _, phi, hamiltonian, saddle_iter, error_level, elapsed_time, is_net_failed = \
                self.find_saddle_point(w_aux=w_aux, tolerance=self.saddle["tolerance"], net=self.net)

            # # # Update exchange mapping for given chiN set
            # # self.initialize_exchange_mapping()

            # Update timers
            for item in elapsed_time:
                total_elapsed_time[item] += elapsed_time[item]

            total_saddle_iter += saddle_iter
            total_error_level += error_level
            if (is_net_failed): total_net_failed += 1
            
            # If the tolerance of the saddle point was not met, regenerate Langevin random noise and continue
            if np.isnan(error_level) or error_level >= self.saddle["tolerance"]:
                if successive_fail_count < 5:                
                    print("The tolerance of the saddle point was not met. Langevin random noise is regenerated.")

                    # Restore w_aux and phi
                    w_aux = w_aux_copy
                    phi = phi_copy
                    
                    # Increment counts and continue
                    successive_fail_count += 1
                    saddle_fail_count += 1
                    continue
                else:
                    print("The tolerance of the saddle point was not met %d times in a row. Simulation is aborted." % (successive_fail_count))
                    return total_saddle_iter/langevin_step, total_error_level/langevin_step
            else:
                successive_fail_count = 0

            if use_wtmd:
                # Store current order parameter and dH/dχN for updating statistics, e.g., U(Ψ), U'(Ψ) and I1
                dH = self.mpt.compute_h_deriv_chin(self.chi_n, w_aux)
                self.wtmd.store_order_parameter(psi, dH)

                # Update WTMD bias
                if langevin_step % self.wtmd.update_freq == 0:
                    self.wtmd.update_statistics()

                # Write WTMD data
                if langevin_step % self.wtmd.recording_period == 0:
                    self.wtmd.write_data(os.path.join(self.recording["dir"], prefix + "statistics_%06d.mat" % (langevin_step)))

            # Compute H and dH/dχN
            if langevin_step % self.recording["sf_computing_period"] == 0:
                H_history.append(hamiltonian)
                dH = self.mpt.compute_h_deriv_chin(self.chi_n, w_aux)
                for key in self.chi_n:
                    dH_history[key].append(dH[key])

            # Save H and dH/dχN
            if langevin_step % self.recording["sf_recording_period"] == 0:
                H_history = np.array(H_history)
                mdic = {"H_history": H_history}
                for key in self.chi_n:
                    dH_history[key] = np.array(dH_history[key])
                    monomer_pair = sorted(key.split(","))
                    mdic["dH_history_" + monomer_pair[0] + "_" + monomer_pair[1]] = dH_history[key]
                savemat(os.path.join(self.recording["dir"], prefix + "dH_%06d.mat" % (langevin_step)), mdic, long_field_names=True, do_compression=True)
                # Reset dictionary
                H_history = []
                for key in self.chi_n:
                    dH_history[key] = []
                    
            # Calculate structure function
            if langevin_step % self.recording["sf_computing_period"] == 0:
                # Perform Fourier transforms
                mu_fourier = {}
                phi_fourier = {}
                for i in range(S):
                    key = self.monomer_types[i]
                    phi_fourier[key] = np.fft.rfftn(np.reshape(phi[self.monomer_types[i]], self.cb.get_nx()))/self.cb.get_n_grid()
                    mu_fourier[key] = np.zeros_like(phi_fourier[key], np.complex128)
                    for k in range(S-1) :
                        mu_fourier[key] += np.fft.rfftn(np.reshape(w_aux[k], self.cb.get_nx()))*self.mpt.matrix_a_inv[k,i]/self.mpt.eigenvalues[k]/self.cb.get_n_grid()
                # Accumulate S_ij(K), assuming that <u(k)>*<phi(-k)> is zero
                for key in sf_average:
                    monomer_pair = sorted(key.split(","))
                    sf_average[key] += mu_fourier[monomer_pair[0]]* np.conj( phi_fourier[monomer_pair[1]])

            # Save structure function
            if langevin_step % self.recording["sf_recording_period"] == 0:
                # Make a dictionary for chi_n
                chi_n_mat = {}
                for key in self.chi_n:
                    monomer_pair = sorted(key.split(","))
                    chi_n_mat[monomer_pair[0] + "," + monomer_pair[1]] = self.chi_n[key]
                mdic = {"dim":self.cb.get_dim(), "nx":self.cb.get_nx(), "lx":self.cb.get_lx(),
                        "chi_n":chi_n_mat, "chain_model":self.chain_model, "ds":self.ds,
                        "dt":self.langevin["dt"], "nbar":self.langevin["nbar"], "initial_params":self.params}
                # Add structure functions to the dictionary
                for key in sf_average:
                    sf_average[key] *= self.recording["sf_computing_period"]/self.recording["sf_recording_period"]* \
                            self.cb.get_volume()*np.sqrt(self.langevin["nbar"])
                    monomer_pair = sorted(key.split(","))
                    mdic["structure_function_" + monomer_pair[0] + "_" + monomer_pair[1]] = sf_average[key]
                savemat(os.path.join(self.recording["dir"], prefix + "structure_function_%06d.mat" % (langevin_step)), mdic, long_field_names=True, do_compression=True)
                # Reset arrays
                for key in sf_average:
                    sf_average[key][:,:,:] = 0.0

            # Save simulation data
            if langevin_step % self.recording["recording_period"] == 0:
                w = self.mpt.to_monomer_fields(w_aux)
                self.save_simulation_data(
                    path=os.path.join(self.recording["dir"], "fields_%06d.mat" % (langevin_step)),
                    w=w, phi=phi, langevin_step=langevin_step, normal_noise_prev=normal_noise_prev)

        # Save final configuration
        w = self.mpt.to_monomer_fields(w_aux)
        self.save_simulation_data(path=final_fields_configuration_file_name, 
            w=w, phi=phi, langevin_step=0, 
            normal_noise_prev=normal_noise_prev)

        # estimate execution time
        time_duration = time.time() - time_start
        
        sum_total_elapsed_time = 0.0
        for item in total_elapsed_time:
            sum_total_elapsed_time += total_elapsed_time[item]
            
        print("\nTotal elapsed time: %f, Elapsed time per Langevin step: %f" %
            (time_duration, time_duration/(max_step+1-start_langevin_step)))
        print("Total iterations for saddle points: %d, Iterations per Langevin step: %f" %
            (total_saddle_iter, total_saddle_iter/(max_step+1-start_langevin_step)))
        print("Elapsed time ratio:")
        print("\tPseudo-spectral solver: %f" % (total_elapsed_time["solver"]/time_duration))
        print("\tDeep learning : %f" % (total_elapsed_time["neural_net"]/time_duration))
        print("\tAnderson mixing: %f" % (total_elapsed_time["am"]/time_duration))
        print("\tPolymer concentrations: %f" % (total_elapsed_time["phi"]/time_duration))
        print("\tHamiltonian: %f" % (total_elapsed_time["hamiltonian"]/time_duration))
        print("\tDerivatives of Hamiltonian w.r.t. imaginary fields: %f" % (total_elapsed_time["h_deriv"]/time_duration))
        print("\tLangevin dynamics: %f" % (total_elapsed_time["langevin"]/time_duration))
        print("\tOther computations on Python: %f" % (1.0 - sum_total_elapsed_time/time_duration))
        print( "The number of times that tolerance of saddle point was not met and Langevin random noise was regenerated: %d times" % 
            (saddle_fail_count))
        print( "The number of times that the neural network could not reduce the incompressibility error (or saddle point error) and switched to Anderson mixing: %d times" % 
            (total_net_failed))
        return total_saddle_iter/(max_step+1-start_langevin_step), total_error_level/(max_step+1-start_langevin_step)

    def find_saddle_point(self, w_aux, tolerance, net=None):

        # The number of components
        S = len(self.monomer_types)

        # The numbers of real and imaginary fields respectively
        R = self.R
        I = self.I
        
        # Assign large initial value for error
        error_level = 1e20

        # Reset Anderson mixing module
        self.am.reset_count()

        # Init timers
        elapsed_time = {}
        elapsed_time["neural_net"] = 0.0
        elapsed_time["solver"] = 0.0
        elapsed_time["phi"] = 0.0
        elapsed_time["am"] = 0.0
        elapsed_time["hamiltonian"] = 0.0
        elapsed_time["h_deriv"] = 0.0
        is_net_failed = False

        # Saddle point iteration begins here
        for saddle_iter in range(1,self.saddle["max_iter"]+1):

            # Compute total concentrations with noised w_aux
            phi, elapsed_time_phi = self.compute_concentrations(w_aux)

            # Compute functional derivatives of Hamiltonian w.r.t. imaginary fields 
            h_deriv, elapsed_time_deriv = self.mpt.compute_func_deriv(w_aux, phi, self.mpt.aux_fields_imag_idx)

            # Update elapsed_time
            for item in elapsed_time_phi:
                elapsed_time[item] += elapsed_time_phi[item]
            for item in elapsed_time_deriv:
                elapsed_time[item] += elapsed_time_deriv[item]

            # Compute total error
            old_error_level = error_level
            error_level_array = np.std(h_deriv, axis=1)
            error_level = np.max(error_level_array)

            # Print iteration # and error levels
            if(self.verbose_level == 2 or self.verbose_level == 1 and
            (error_level < tolerance or saddle_iter == self.saddle["max_iter"])):

                # Calculate Hamiltonian
                time_h_start = time.time()
                total_partitions = [self.solver.get_total_partition(p) for p in range(self.molecules.get_n_polymer_types())]
                hamiltonian = self.mpt.compute_hamiltonian(self.molecules, w_aux, total_partitions)
                elapsed_time["hamiltonian"] += time.time() - time_h_start

                # Check the mass conservation
                mass_error = np.mean(h_deriv[I-1])
                print("%8d %12.3E " % (saddle_iter, mass_error), end=" [ ")
                for p in range(self.molecules.get_n_polymer_types()):
                    print("%13.7E " % (self.solver.get_total_partition(p)), end=" ")
                print("] %15.9f   [" % (hamiltonian), end="")
                for i in range(I):
                    print("%13.7E" % (error_level_array[i]), end=" ")
                print("]")

            # When neural net fails
            if net and is_net_failed == False and (error_level >= old_error_level or np.isnan(error_level)):
                # Restore fields from backup
                w_aux[self.mpt.aux_fields_imag_idx] = w_imag_backup
                is_net_failed = True
                print("%8d This neural network could not reduce the incompressibility error (or saddle point error), and it was switched to Anderson mixing." % (saddle_iter))
                continue

            # Conditions to end the iteration
            if error_level < tolerance:
                break

            if net and not is_net_failed:
                time_d_start = time.time()
                # Make a backup of imaginary fields
                w_imag_backup = w_aux[self.mpt.aux_fields_imag_idx].copy()
                # Make an array of real fields
                w_real = w_aux[self.mpt.aux_fields_real_idx]
                # Predict field difference using neural network
                w_imag_diff = net.predict_w_imag(w_real, h_deriv, self.cb.get_nx())
                # Update fields
                w_aux[self.mpt.aux_fields_imag_idx] += w_imag_diff
                elapsed_time["neural_net"] += time.time() - time_d_start
            else:
                
                # Scaling h_deriv
                for count, i in enumerate(self.mpt.aux_fields_imag_idx):
                    h_deriv[count] *= self.dt_scaling[i]
                
                # Calculate new fields using simple and Anderson mixing
                time_a_start = time.time()
                w_aux[self.mpt.aux_fields_imag_idx] = \
                    np.reshape(self.am.calculate_new_fields(w_aux[self.mpt.aux_fields_imag_idx],
                    -h_deriv, old_error_level, error_level), [I, self.cb.get_n_grid()])
                elapsed_time["am"] += time.time() - time_a_start

        # Set mean of pressure field to zero
        w_aux[S-1] -= np.mean(w_aux[S-1])

        return w_aux[self.mpt.aux_fields_imag_idx], \
            phi, hamiltonian, saddle_iter, error_level, elapsed_time, is_net_failed
