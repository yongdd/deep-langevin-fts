import os
import time
import glob
import shutil
import pathlib
import numpy as np
import itertools
from scipy.io import savemat, loadmat

class WTMDMulti:
    def __init__(self, nx, lx):
        # Well-tempered metadynamics
        # [*J. Chem. Phys.* **2022**, 157, 114902]
        self.ell=4
        self.kc=6.02
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

        m = cb.get_nx()
        L = cb.get_lx()
        M = m[0]*m[1]*m[2]
        Mk = m[0]*m[1]*(m[2]//2+1)
        V = L[0]*L[1]*L[2]
        
        K = np.zeros(Mk)
        wt = np.zeros(Mk)
        fk = np.zeros(Mk)
        
        wt[:] = 2.0
        for k0 in range(-(m[0]-1)//2, m[0]//2+1):
            if k0<0:
                K0 = k0+m[0]
            else:
                K0 = k0
            kx_sq = k0*k0/(L[0]*L[0])
            for k1 in range(-(m[1]-1)//2, m[1]//2+1):
                if k1<0:
                    K1 = k1+m[1]
                else:
                    K1 = k1
                ky_sq = k1*k1/(L[1]*L[1])
                for k2 in range(0, m[2]//2+1):
                    kz_sq = k2*k2/(L[2]*L[2])
                    k = k2+(m[2]//2+1)*(K1+m[1]*K0)
                    K[k] = 2*np.pi*np.power(kx_sq+ky_sq+kz_sq, 0.5)
                    if k2==0 or k2==m[2]//2:
                        wt[k]=1
                    fk[k] = 1.0/(1.0 + np.exp(12.0*(K[k]-self.kc)/self.kc))

        self.wt = np.reshape(wt, (m[0],m[1],m[2]//2+1))
        self.fk = np.reshape(fk, (m[0],m[1],m[2]//2+1))
        
    # Compute order parameter for WTMD
    def get_psi(self, w_exchange_real_k):
        
        Psi = 0.0
        for count, i in enumerate(self.exchange_fields_real_idx):
            Psi += np.sum(np.power(np.absolute(w_exchange_real_k[count]), self.ell)*self.fk*self.wt)
        Psi = np.power(Psi/self.R, 1.0/self.ell)/self.cb.get_n_grid()
        return Psi

    def get_bias(self, Psi, w_exchange_real_k):
        
        # Calculate current value of U'(Psi)
        x = (Psi-self.Psi_min)/self.dPsi
        i = np.floor(x).astype(int)
        x = x-i
        up_hat = (1.0-x)*self.up[i] + x*self.up[i+1]

        N = 1.0/self.ds
        M = self.cb.get_n_grid()
        V = self.cb.get_volume()
        dPsi_dwk = np.zeros_like(w_exchange_real_k)
        dPsi_dwr = np.zeros([self.R] + list(self.cb.get_nx()))
        for count, i in enumerate(self.exchange_fields_real_idx):
            # Calculate derivative of order parameter with respect to wk
            dPsi_dwk[count] = np.power(np.absolute(w_exchange_real_k[count]),self.ell-2.0)*\
                np.power(Psi,1.0-self.ell)*w_exchange_real_k[count]*self.fk/self.R
            # Calculate derivative of order parameter with respect to w
            dPsi_dwr[count] = np.fft.irfftn(dPsi_dwk[count], self.cb.get_nx())*np.power(M, 2.0-self.ell)*N/V
            print("np.std(dPsi_dwr[count])", np.std(dPsi_dwr[count]))

        return np.reshape(V/N*up_hat*dPsi_dwr, (self.R, M))

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
        # for count, i in enumerate(self.exchange_fields_real_idx):
        #     w_exchange_real_k[count] = np.fft.rfftn(np.reshape(w_exchange[i], self.cb.get_nx()))
        #     print("count, i, np.std(w_exchange[i]), np.std(w_lambda[i]:", count, i, np.std(w_exchange[i]), np.std(w_lambda[i])) 
        # Psi = self.get_psi(w_exchange_real_k)
        # bias = self.get_bias(Psi, w_exchange_real_k)
        # for count, i in enumerate(self.exchange_fields_real_idx):
        #     print("Psi, np.std(w_lambda[count]), np.std(bias): ", Psi, np.std(w_lambda[count]), np.std(bias[count]))
        # w_lambda += bias

        # for count, i in enumerate(self.exchange_fields_real_idx):
        #     scale_lambda = np.abs(self.exchange_eigenvalues[i])/np.max(np.abs(self.exchange_eigenvalues))
        #     print("count, i, scale_lambda, np.sqrt(scale_lambda)", count, i, scale_lambda, np.sqrt(scale_lambda))
        #     w_exchange[i] += -w_lambda[count]*self.langevin["dt"]*scale_lambda+ \
        #             (normal_noise_prev[count] + normal_noise_current[count])*np.sqrt(scale_lambda) /2