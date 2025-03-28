import os
import time
import glob
import shutil
import pathlib
import numpy as np
import itertools
from scipy.io import savemat, loadmat

class WTMDMax:
    def __init__(self, nx, lx):
        # Well-tempered metadynamics
        # [*J. Chem. Phys.* **2022**, 157, 114902]
        self.sigma_Psi=40.0/250
        self.DT=5.0,
        self.Psi_min=-1.0
        self.DIM2=5000
        self.dPsi=0.5/250
        self.update_freq=1000
        
        self.u = np.zeros(self.DIM2)
        self.up = np.zeros(self.DIM2)
        self.I0 = np.zeros(self.DIM2)
        self.I1 = np.zeros(self.DIM2)
        
    # Compute order parameter for WTMD
    def get_Psi(self, w_minus_k):
        
        w_minus_squared = np.sqrt((w_minus_k*np.conj(w_minus_k)).real)/self.cb.get_total_grid()
        w_minus_max = np.max(w_minus_squared)
        w_minus_argmax = np.argmax(w_minus_squared)
        return w_minus_max, w_minus_argmax

    def get_bias(self, Psi, argmax, wk):
        
        # Calculate current value of U'(Psi)
        x = (Psi-self.Psi_min)/self.dPsi
        i = np.floor(x).astype(int)
        x = x-i
        up_hat = (1.0-x)*self.up[i] + x*self.up[i+1]

        # Calculate derivative of order parameter with respect to w
        nx = self.cb.get_nx()
        lx = self.cb.get_lx()
        
        space_x, space_y, space_z = np.meshgrid(
            lx[0]*np.arange(nx[0])/nx[0],
            lx[1]*np.arange(nx[1])/nx[1],
            lx[2]*np.arange(nx[2])/nx[2], indexing='ij')

        space_kx, space_ky, space_kz = np.meshgrid(
            2*np.pi/lx[0]*np.arange(nx[0]),
            2*np.pi/lx[1]*np.arange(nx[1]),
            2*np.pi/lx[2]*np.arange(nx[2]//2+1), indexing='ij')

        space_kx = space_kx.flatten()
        space_ky = space_ky.flatten()
        space_kz = space_kz.flatten()

        kx_star = space_kx[argmax]
        ky_star = space_ky[argmax]
        kz_star = space_kz[argmax]
        
        N = 1.0/self.ds
        M = self.cb.get_total_grid()
        V = self.cb.get_volume()
        
        dPsi_dwr = 0.5/Psi*wk.flatten()[argmax]/M*np.exp(1.0j*(space_x*kx_star+space_y*ky_star+space_z*kz_star))/V
        dPsi_dwr = 2*dPsi_dwr.real
        print("np.std(dPsi_dwr)", np.std(dPsi_dwr))

        return np.reshape(V*up_hat*dPsi_dwr, self.cb.get_total_grid())

    def update_bias(self, Psi_hat, w_exchange):
        
        # Compute amplitude and gaussian
        CV = np.sqrt(self.langevin["nbar"])*self.cb.get_volume()
        # CV = 1.0
        w2_hat = np.mean(w_exchange[0]*w_exchange[0])
        Psi = self.Psi_min + np.arange(self.DIM2)*self.dPsi
        TEMP = (Psi_hat-Psi)/self.sigma_Psi
        amplitude = np.exp(-CV*self.u/self.DT)/CV
        gaussian = np.exp(-0.5*TEMP*TEMP)
        
        # Update u, up, I0, I1
        self.u  += amplitude*gaussian
        self.up += (TEMP/self.sigma_Psi-CV*self.up/self.DT)*amplitude*gaussian
        self.I0 += gaussian
        self.I1 += w2_hat*gaussian
        
        print("np.max(np.abs(amplitude)), np.max(np.abs(gaussian))", np.max(np.abs(amplitude)), np.max(np.abs(gaussian)))

        # # Compute bias fields for WTMD
        # wk = np.fft.rfftn(np.reshape(w_exchange[0]-np.mean(w_exchange[0]), self.cb.get_nx()))
        # Psi, Psi_argmax = self.get_Psi(wk)
        # bias = self.get_bias(Psi, Psi_argmax, wk)
        # print("Psi, argmax, np.std(w_lambda[0]), np.std(bias): ", Psi, Psi_argmax, np.std(w_lambda[0]), np.std(bias))             
        # w_lambda[0] += bias