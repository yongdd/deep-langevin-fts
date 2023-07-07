import os
import time
import glob
import shutil
import pathlib
import numpy as np
import itertools
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
# from model.atr_cas_mish import *       # LitAtrousCascadeMish, 
# from model.atr_cas_x import *        # LitAtrousCascadeXception, 

# OpenMP environment variables
os.environ["MKL_NUM_THREADS"] = "1"  # always 1
os.environ["OMP_STACKSIZE"] = "1G"

class TrainAndInference(pl.LightningModule):
    def __init__(self, dim, in_channels=3, mid_channels=32, out_channels=1, kernel_size = 3, lr=None, epoch_offset=None):
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
        torch.save(self.state_dict(), os.path.join(path, 'epoch_%d.pth' % (self.current_epoch + self.epoch_offset)))
      
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
                output[0,i,:,:,] *= total_std_h_deriv[i]*20

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
            Y[i,:] /= std_h_deriv*20


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
        #    print(self.file_list[data_idx], i, torch.std(Y[2*i]), torch.mean(Y[2*i+1]))
        
        return {
            'input' : X.to(dtype=torch.float32),
            'target': Y.to(dtype=torch.float32)
        }

def calculate_sigma(langevin_nbar, langevin_dt, n_grids, volume):
        return np.sqrt(2*langevin_dt*n_grids/(volume*np.sqrt(langevin_nbar)))

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

        # (c++ class) Create a factory for given platform and chain_model
        if "reduce_gpu_memory_usage" in params and platform == "cuda":
            factory = PlatformSelector.create_factory(platform, params["chain_model"], params["reduce_gpu_memory_usage"])
        else:
            factory = PlatformSelector.create_factory(platform, params["chain_model"], False)
        factory.display_info()

        # (C++ class) Computation box
        cb = factory.create_computation_box(params["nx"], params["lx"])

        # Flory-Huggins parameters, chi*N
        self.chi_n = {}
        for pair_chi_n in params["chi_n"]:
            assert(pair_chi_n[0] in params["segment_lengths"]), \
                f"Monomer type '{pair_chi_n[0]}' is not in 'segment_lengths'."
            assert(pair_chi_n[1] in params["segment_lengths"]), \
                f"Monomer type '{pair_chi_n[1]}' is not in 'segment_lengths'."
            assert(len(set(pair_chi_n[0:2])) == 2), \
                "Do not add self interaction parameter, " + str(pair_chi_n[0:3]) + "."
            assert(not frozenset(pair_chi_n[0:2]) in self.chi_n), \
                f"There are duplicated chi N ({pair_chi_n[0:2]}) parameters."
            self.chi_n[frozenset(pair_chi_n[0:2])] = pair_chi_n[2]

        for monomer_pair in itertools.combinations(self.monomer_types, 2):
            if not frozenset(list(monomer_pair)) in self.chi_n:
                self.chi_n[frozenset(list(monomer_pair))] = 0.0

        for monomer_pair in itertools.combinations(self.monomer_types, 2):
            if not frozenset(list(monomer_pair)) in self.chi_n:
                self.chi_n[frozenset(list(monomer_pair))] = 0.0
                

        # Exchange mapping matrix.
        # See paper *J. Chem. Phys.* **2014**, 141, 174103
        S = len(self.monomer_types)
        self.matrix_o = np.zeros((S-1,S-1))
        self.matrix_a = np.zeros((S,S))
        self.matrix_a_inv = np.zeros((S,S))
        self.vector_s = np.zeros(S-1)

        for i in range(S-1):
            key = frozenset([self.monomer_types[i], self.monomer_types[S-1]])
            self.vector_s[i] = self.chi_n[key]

        matrix_chi = np.zeros((S,S))
        matrix_chin = np.zeros((S-1,S-1))

        for i in range(S):
            for j in range(i+1,S):
                key = frozenset([self.monomer_types[i], self.monomer_types[j]])
                if key in self.chi_n:
                    matrix_chi[i,j] = self.chi_n[key]
                    matrix_chi[j,i] = self.chi_n[key]
        
        for i in range(S-1):
            for j in range(S-1):
                matrix_chin[i,j] = matrix_chi[i,j] - matrix_chi[i,S-1] - matrix_chi[j,S-1] # fix a typo in the paper

        self.matrix_chi = matrix_chi

        self.exchange_eigenvalues, self.matrix_o = np.linalg.eig(matrix_chin)
        
        # Indices whose exchange fields are real
        self.exchange_fields_real_idx = []
        # Indices whose exchange fields are imaginary including the pressure field
        self.exchange_fields_imag_idx = []
        for i in range(S-1):
            assert(not np.isclose(self.exchange_eigenvalues[i], 0.0)), \
                "One of eigenvalues is zero. change your chin values."
            if self.exchange_eigenvalues[i] > 0:
                self.exchange_fields_imag_idx.append(i)
            else:
                self.exchange_fields_real_idx.append(i)
        self.exchange_fields_imag_idx.append(S-1) # add pressure field

        # The numbers of real and imaginary fields, respectively
        self.R = len(self.exchange_fields_real_idx)
        self.I = len(self.exchange_fields_imag_idx)

        if self.I > 1:
            print("(Warning!) For a given chi N interaction parameter set, at least one of the exchange fields is an imaginary field. ", end="")
            print("The field fluctuations would not be fully reflected. Run this simulation at your own risk.")
        
        # Matrix A and Inverse for converting between exchange fields and species chemical potential fields
        self.matrix_a[0:S-1,0:S-1] = self.matrix_o[0:S-1,0:S-1]
        self.matrix_a[:,S-1] = 1
        self.matrix_a_inv[0:S-1,0:S-1] = np.transpose(self.matrix_o[0:S-1,0:S-1])
        for i in range(S-1):
            self.matrix_a_inv[i,S-1] =  -np.sum(self.matrix_o[:,i])
            self.matrix_a_inv[S-1,S-1] = 1

        # Total volume fraction
        assert(len(params["distinct_polymers"]) >= 1), \
            "There is no polymer chain."

        total_volume_fraction = 0.0
        for polymer in params["distinct_polymers"]:
            total_volume_fraction += polymer["volume_fraction"]
        assert(np.isclose(total_volume_fraction,1.0)), "The sum of volume fraction must be equal to 1."

        # Polymer Chains
        self.random_fraction = {}
        for polymer_counter, polymer in enumerate(params["distinct_polymers"]):
            block_length_list = []
            block_monomer_type_list = []
            v_list = []
            u_list = []

            alpha = 0.0             # total_relative_contour_length
            block_count = 0
            is_linear_chain = not "v" in polymer["blocks"][0]
            for block in polymer["blocks"]:
                block_length_list.append(block["length"])
                block_monomer_type_list.append(block["type"])
                alpha += block["length"]

                if is_linear_chain:
                    assert(not "v" in block), \
                        "Index v should exist in all blocks, or it should not exist in all blocks for each polymer." 
                    assert(not "u" in block), \
                        "Index u should exist in all blocks, or it should not exist in all blocks for each polymer." 

                    v_list.append(block_count)
                    u_list.append(block_count+1)
                else:
                    assert("v" in block), \
                        "Index v should exist in all blocks, or it should not exist in all blocks for each polymer." 
                    assert("u" in block), \
                        "Index u should exist in all blocks, or it should not exist in all blocks for each polymer." 

                    v_list.append(block["v"])
                    u_list.append(block["u"])
                block_count += 1

            polymer.update({"block_monomer_types":block_monomer_type_list})
            polymer.update({"block_lengths":block_length_list})
            polymer.update({"v":v_list})
            polymer.update({"u":u_list})

        # Random Copolymer Chains
        for polymer in params["distinct_polymers"]:

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
                statistical_segment_length += params["segment_lengths"][monomer_type]**2 * polymer["blocks"][0]["fraction"][monomer_type]
                total_random_fraction += polymer["blocks"][0]["fraction"][monomer_type]
            statistical_segment_length = np.sqrt(statistical_segment_length)

            assert(np.isclose(total_random_fraction, 1.0)), \
                "The sum of volume fraction of random copolymer must be equal to 1."

            random_type_string = polymer["blocks"][0]["type"]
            assert(not random_type_string in params["segment_lengths"]), \
                f"The name of random copolymer '{random_type_string}' is already used as a type in 'segment_lengths' or other random copolymer"

            # Add random copolymers
            polymer["block_monomer_types"] = [random_type_string]
            params["segment_lengths"].update({random_type_string:statistical_segment_length})
            self.random_fraction[random_type_string] = polymer["blocks"][0]["fraction"]

        # (C++ class) Mixture box
        if "use_superposition" in params:
            mixture = factory.create_mixture(params["ds"], params["segment_lengths"], params["use_superposition"])
        else:
            mixture = factory.create_mixture(params["ds"], params["segment_lengths"], True)

        # Add polymer chains
        for polymer in params["distinct_polymers"]:
            # print(polymer["volume_fraction"], polymer["block_monomer_types"], polymer["block_lengths"], polymer["v"], polymer["u"])
            mixture.add_polymer(polymer["volume_fraction"], polymer["block_monomer_types"], polymer["block_lengths"], polymer["v"] ,polymer["u"])

        # Langevin dynamics
        # standard deviation of normal noise
        langevin_sigma = calculate_sigma(params["langevin"]["nbar"], params["langevin"]["dt"], np.prod(params["nx"]), np.prod(params["lx"]))

        # Set random generator
        if random_seed == None:         
            self.random_bg = np.random.PCG64()  # Set random bit generator
        else:
            self.random_bg = np.random.PCG64(random_seed)
        self.random = np.random.Generator(self.random_bg)

        # -------------- print simulation parameters ------------
        print("---------- Simulation Parameters ----------")
        print("Platform :", "cuda")
        print("Statistical Segment Lengths:", params["segment_lengths"])
        print("Box Dimension: %d" % (cb.get_dim()))
        print("Nx:", cb.get_nx())
        print("Lx:", cb.get_lx())
        print("dx:", cb.get_dx())
        print("Volume: %f" % (cb.get_volume()))

        print("Chain model: %s" % (params["chain_model"]))
        print("Segment lengths:\n\t", list(params["segment_lengths"].items()))
        print("Conformational asymmetry (epsilon): ")
        for monomer_pair in itertools.combinations(self.monomer_types,2):
            print("\t%s/%s: %f" % (monomer_pair[0], monomer_pair[1], params["segment_lengths"][monomer_pair[0]]/params["segment_lengths"][monomer_pair[1]]))

        print("chiN: ")
        for pair in self.chi_n:
            print("\t%s, %s: %f" % (list(pair)[0], list(pair)[1], self.chi_n[pair]))

        print("Eigenvalues:\n\t", self.exchange_eigenvalues)
        print("Column eigenvectors:\n\t", str(self.matrix_o).replace("\n", "\n\t"))
        print("Vector chi_iS:\n\t", str(self.vector_s).replace("\n", "\n\t"))
        print("Mapping matrix A:\n\t", str(self.matrix_a).replace("\n", "\n\t"))
        print("Inverse of A:\n\t", str(self.matrix_a_inv).replace("\n", "\n\t"))
        print("A*Inverse[A]:\n\t", str(np.matmul(self.matrix_a, self.matrix_a_inv)).replace("\n", "\n\t"))
        print("Imaginary Fields", self.exchange_fields_imag_idx)

        for p in range(mixture.get_n_polymers()):
            print("distinct_polymers[%d]:" % (p) )
            print("\tvolume fraction: %f, alpha: %f, N: %d" %
                (mixture.get_polymer(p).get_volume_fraction(),
                 mixture.get_polymer(p).get_alpha(),
                 mixture.get_polymer(p).get_n_segment_total()))

        print("Invariant Polymerization Index (N_Ref): %d" % (params["langevin"]["nbar"]))
        print("Langevin Sigma: %f" % (langevin_sigma))
        print("Random Number Generator: ", self.random_bg.state)

        mixture.display_blocks()
        mixture.display_propagators()

        #  Save Internal Variables
        self.params = params
        self.chain_model = params["chain_model"]
        self.chi_n = params["chi_n"]
        self.ds = params["ds"]
        self.langevin = params["langevin"].copy()
        self.langevin.update({"sigma":langevin_sigma})
        self.langevin.pop("max_step", None)

        self.verbose_level = params["verbose_level"]
        self.saddle = params["saddle"].copy()
        self.recording = params["recording"].copy()

        self.factory = factory
        self.cb = cb
        self.mixture = mixture
        self.pseudo = None
        self.am = None
        
    def create_solvers(self):
        
        params = self.params
        
        # (C++ class) Solver using Pseudo-spectral method
        self.pseudo = self.factory.create_pseudo(self.cb, self.mixture)

        # (C++ class) Fields relaxation using Anderson Mixing
        self.am = self.factory.create_anderson_mixing(
            len(self.exchange_fields_imag_idx)*np.prod(params["nx"]),   # the number of variables
            params["am"]["max_hist"],                                   # maximum number of history
            params["am"]["start_error"],                                # when switch to AM from simple mixing
            params["am"]["mix_min"],                                    # minimum mixing rate of simple mixing
            params["am"]["mix_init"])                                   # initial mixing rate of simple mixing
        
    def save_training_data(self, path, w_real, h_deriv, w_diff):

        # Make dictionary for chi_n
        chi_n_mat = {}
        for pair_chi_n in self.params["chi_n"]:
            sorted_name_pair = sorted(pair_chi_n[0:2])
            chi_n_mat[sorted_name_pair[0] + "," + sorted_name_pair[1]] = pair_chi_n[2]

        np.savez(path,
            dim=self.cb.get_dim(), nx=self.cb.get_nx(), lx=self.cb.get_lx(),
            chi_n=chi_n_mat, chain_model=self.chain_model, ds=self.ds,
            dt=self.langevin["dt"], nbar=self.langevin["nbar"], params=self.params,
            w_real=w_real.astype(np.float16),
            h_deriv=h_deriv.astype(np.float16),
            w_diff=w_diff.astype(np.float16))

    def save_simulation_data(self, path, w, phi, langevin_step, normal_noise_prev):
        
        # Make dictionary for w fields
        w_species = {}
        for i, name in enumerate(self.monomer_types):
            w_species[name] = w[i]

        # Make dictionary for chi_n
        chi_n_mat = {}
        for pair_chi_n in self.params["chi_n"]:
            sorted_name_pair = sorted(pair_chi_n[0:2])
            chi_n_mat[sorted_name_pair[0] + "," + sorted_name_pair[1]] = pair_chi_n[2]

        # Make dictionary for data
        mdic = {"dim":self.cb.get_dim(), "nx":self.cb.get_nx(), "lx":self.cb.get_lx(),
            "chi_n":chi_n_mat, "chain_model":self.chain_model, "ds":self.ds,
            "dt":self.langevin["dt"], "nbar":self.langevin["nbar"], "initial_params": self.params,
            "langevin_step":langevin_step,
            "random_generator":self.random_bg.state["bit_generator"],
            "random_state_state":str(self.random_bg.state["state"]["state"]),
            "random_state_inc":str(self.random_bg.state["state"]["inc"]),
            "normal_noise_prev":normal_noise_prev,
            "w": w_species, "phi":phi, "monomer_types":self.monomer_types}
        
        # Save data with matlab format
        savemat(path, mdic)

    def make_training_data(self, initial_fields, last_training_step_file_name):

        # Skip if this process is not the primary process of DPP in Pytorch-Lightning.
        # This is because DDP duplicates the main script when using multiple GPUs.
        if self.is_secondary:
            return

        # Create pseudo and anderson mixing solvers if necessary
        if type(self.pseudo) == type(None):
            self.create_solvers()

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

        # Exchange-mapped chemical potential fields
        w_exchange = np.matmul(self.matrix_a_inv, w)

        # Find saddle point 
        _, phi, _, _, _, _ = self.find_saddle_point(w_exchange=w_exchange, tolerance=self.saddle["tolerance"], net=None)

        # Create an empty array for field update algorithm
        normal_noise_prev = np.zeros([R, self.cb.get_n_grid()], dtype=np.float64)

        #------------------ run ----------------------
        print("---------- Collect Training Data ----------")
        print("iteration, mass error, total partitions, total energy, incompressibility error (or saddle point error)")
        for langevin_step in range(1, self.training["max_step"]+1):
            print("Langevin step: ", langevin_step)

            # Update w_minus using Leimkuhler-Matthews method
            normal_noise_current = self.random.normal(0.0, self.langevin["sigma"], [R, self.cb.get_n_grid()])
            w_lambda = np.zeros([R, self.cb.get_n_grid()], dtype=np.float64) # array for output fields
            
            for count, i in enumerate(self.exchange_fields_real_idx):
                w_lambda[count] -= 1.0/self.exchange_eigenvalues[i]*w_exchange[i]
            for count, i in enumerate(self.exchange_fields_real_idx):
                for j in range(S-1):
                    w_lambda[count] += 1.0/self.exchange_eigenvalues[i]*self.matrix_o[j,i]*self.vector_s[j]
                    w_lambda[count] += self.matrix_o[j,i]*phi[self.monomer_types[j]]

            w_exchange[self.exchange_fields_real_idx] += -w_lambda*self.langevin["dt"] + (normal_noise_prev + normal_noise_current)/2

            # Swap two noise arrays
            normal_noise_prev, normal_noise_current = normal_noise_current, normal_noise_prev
            
            # Find saddle point
            w_imag_start = w_exchange[self.exchange_fields_imag_idx].copy()
            w_imag, phi, _, _, _, _ = self.find_saddle_point(w_exchange=w_exchange, tolerance=self.saddle["tolerance"], net=None)

            # Training data is sampled from random noise distribution with various standard deviations
            if (langevin_step % self.training["recording_period"] == 0):
                w_imag_tol = w_imag.copy()
                
                # Find more accurate saddle point
                w_imag, _, _, _, _, _ = self.find_saddle_point(w_exchange=w_exchange, tolerance=self.training["tolerance"], net=None)
                w_imag_ref = w_imag.copy()
                
                # Select noise levels between sigma a and sigma b for each imaginary field
                sigma_array = np.zeros([I, self.training["recording_n_data"]])
                for i in range(I):
                    sigma_a = np.std(w_imag_start[i] - w_imag_ref[i])
                    sigma_b = np.std(w_imag_tol[i]   - w_imag_ref[i])
                    log_sigma_sample = np.random.uniform(np.log(sigma_b), np.log(sigma_a), self.training["recording_n_data"])
                    sigma_array[i] = np.exp(log_sigma_sample)

                # print(sigma_array)
                #print(np.log(sigma_list))
                for std_idx in range(self.training["recording_n_data"]):
                    
                    sigma = sigma_array[:,std_idx]
                    print("Standard deviation of the generated noise for each imaginary field", sigma)
                    
                    # Generate random noise
                    w_imag_with_noise = w_imag_ref.copy()
                    random_noise = np.zeros([I, self.cb.get_n_grid()])
                    for i in range(I):
                        random_noise[i] = np.random.normal(0, sigma[i], self.cb.get_n_grid())
                        w_imag_with_noise[i] += random_noise[i]

                    # Add random noise and convert to species chemical potential fields
                    w_exchange_noise = w_exchange.copy()
                    w_exchange_noise[self.exchange_fields_imag_idx] = w_imag_with_noise
                    w = np.matmul(self.matrix_a, w_exchange_noise)

                    # Make a dictionary for input fields 
                    w_input = {}
                    for i in range(S):
                        w_input[self.monomer_types[i]] = w[i]
                    for random_polymer_name, random_fraction in self.random_fraction.items():
                        w_input[random_polymer_name] = np.zeros(self.cb.get_n_grid(), dtype=np.float64)
                        for monomer_type, fraction in random_fraction.items():
                            w_input[random_polymer_name] += w_input[monomer_type]*fraction

                    # For the given fields, compute the polymer statistics
                    self.pseudo.compute_statistics(w_input)

                    # Compute total concentration for each monomer type
                    phi = {}
                    for monomer_type in self.monomer_types:
                        phi[monomer_type] = self.pseudo.get_total_concentration(monomer_type)

                    # Add random copolymer concentration to each monomer type
                    for random_polymer_name, random_fraction in self.random_fraction.items():
                        phi[random_polymer_name] = self.pseudo.get_total_concentration(random_polymer_name)
                        for monomer_type, fraction in random_fraction.items():
                            phi[monomer_type] += phi[random_polymer_name]*fraction

                    # Calculate incompressibility and saddle point error
                    h_deriv = np.zeros([I, self.cb.get_n_grid()], dtype=np.float64)
                    for count, i in enumerate(self.exchange_fields_imag_idx):
                        if i != S-1:
                            h_deriv[count] -= 1.0/self.exchange_eigenvalues[i]*w_exchange_noise[i]
                    for count, i in enumerate(self.exchange_fields_imag_idx):
                        if i != S-1:
                            for j in range(S-1):
                                h_deriv[count] += 1.0/self.exchange_eigenvalues[i]*self.matrix_o[j,i]*self.vector_s[j]
                                h_deriv[count] += self.matrix_o[j,i]*phi[self.monomer_types[j]]
                    for i in range(S):
                        h_deriv[I-1] += phi[self.monomer_types[i]]
                    h_deriv[I-1] -= 1.0

                    # for i in range(I):
                    #     print("%d: %7.2e, %7.2e" % (i, np.std(random_noise[i]), np.std(h_deriv[i])))

                    path = os.path.join(self.training["data_dir"], "%05d_%03d.npz" % (langevin_step, std_idx))
                    print(path)
                    self.save_training_data(path, w_exchange_noise[self.exchange_fields_real_idx], h_deriv, -random_noise)
            
            # Save training check point
            if (langevin_step) % self.recording["recording_period"] == 0:
                w = np.matmul(self.matrix_a, w_exchange)
                self.save_simulation_data(
                    path=os.path.join(self.training["data_dir"], "fields_%06d.mat" % (langevin_step)),
                    w=w, phi=phi, langevin_step=langevin_step,
                    normal_noise_prev=normal_noise_prev)
            
        # Save final configuration to use it as input in actual simulation
        w = np.matmul(self.matrix_a, w_exchange)
        self.save_simulation_data(path=last_training_step_file_name, 
            w=w, phi=phi, langevin_step=0, 
            normal_noise_prev=normal_noise_prev)

    def train_model(self, model_file=None, epoch_offset=None):

        # Free gpu memory for pseudo and anderson mixing
        if type(self.pseudo) != type(None):
            self.pseudo = None
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

        print(f"---------- model file : {model_file} ----------")
        assert((model_file == None and epoch_offset == None) or
             (model_file != None and epoch_offset != None)), \
            "To continue the training, put both model file name and epoch offset."

        # The number of input channels
        in_channels = self.R+2*self.I

        if model_file:                                                    
            self.net = TrainAndInference(dim=self.cb.get_dim(), in_channels=in_channels, mid_channels=features, out_channels=self.I, lr=lr, epoch_offset=epoch_offset+1)
            self.net.load_state_dict(torch.load(model_file), strict=True)
            max_epochs -= epoch_offset+1
        else:
            self.net = TrainAndInference(dim=self.cb.get_dim(), in_channels=in_channels, mid_channels=features, out_channels=self.I, lr=lr)
            
        # Training data
        train_dataset = FtsDataset(data_dir, self.R, self.I)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
        print(len(train_dataset))
        
        # Training
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
        # -------------- deep learning --------------
        saved_weight_dir = self.training["model_dir"]
        torch.set_num_threads(1)

        #-------------- test roughly ------------
        list_saddle_iter_per = []
        file_list = sorted(glob.glob(saved_weight_dir + "/*.pth"), key=lambda l: (len(l), l))
        print(file_list)
        print("iteration, mass error, total_partition, energy_total, error_level")
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

    def continue_run(self, file_name, max_step, model_file=None):

        # Skip if this process is not the primary process of DPP in Pytorch-Lightning.
        # This is because DDP duplicates the main script when using multiple GPUs.
        if self.is_secondary:
            return

        # The number of components
        S = len(self.monomer_types)

        # Load_data
        load_data = loadmat(file_name, squeeze_me=True)

        # Restore random state
        self.random_bg.state ={'bit_generator': 'PCG64',
            'state': {'state': int(load_data["random_state_state"]),
                      'inc':   int(load_data["random_state_inc"])},
                      'has_uint32': 0, 'uinteger': 0}
        print("Restored Random Number Generator: ", self.random_bg.state)

        # Make initial_fields
        initial_fields = {}
        for name in self.monomer_types:
            initial_fields[name] = np.array(load_data["w"][name].tolist())

        # Run
        self.run(initial_fields=initial_fields,
            max_step=max_step, model_file=model_file,
            normal_noise_prev=load_data["normal_noise_prev"],
            start_langevin_step=load_data["langevin_step"]+1)

    def run(self, initial_fields, max_step, model_file=None, normal_noise_prev=None, start_langevin_step=None):

        # ------------ ML Part ------------------
        # Skip if this process is not the primary process of DPP in Pytorch-Lightning.
        # This is because DDP duplicates the main script when using multiple GPUs.
        if self.is_secondary:
            return

        # The number of input channels 
        in_channels = self.R+2*self.I

        # Load deep learning model weights
        print(f"---------- model file : {model_file} ----------")
        if model_file :
            self.net = TrainAndInference(dim=self.cb.get_dim(), in_channels=in_channels, mid_channels=self.training["features"], out_channels=self.I)
            self.net.load_state_dict(torch.load(model_file, map_location =self.device_string), strict=True)
            self.net.set_inference_mode(self.device)

        # ------------ Simulation Part ------------------
        # Create pseudo and anderson mixing solvers if necessary
        if type(self.pseudo) == type(None):
            self.create_solvers()

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
            
        # Exchange-mapped chemical potential fields
        w_exchange = np.matmul(self.matrix_a_inv, w)

        # Find saddle point 
        _, phi, _, _, _, _ = self.find_saddle_point(w_exchange=w_exchange, tolerance=self.saddle["tolerance"], net=None)

        # Arrays for structure function
        sf_average = {} # <u(k) phi(-k)>
        for monomer_id_pair in itertools.combinations_with_replacement(list(range(S)),2):
            sorted_pair = sorted(monomer_id_pair)
            type_pair = self.monomer_types[sorted_pair[0]] + "," + self.monomer_types[sorted_pair[1]]
            sf_average[type_pair] = np.zeros_like(np.fft.rfftn(np.reshape(w[0], self.cb.get_nx())), np.complex128)

        # Create an empty array for field update algorithm
        if type(normal_noise_prev) == type(None) :
            normal_noise_prev = np.zeros([R, self.cb.get_n_grid()], dtype=np.float64)
        else:
            normal_noise_prev = normal_noise_prev

        if start_langevin_step == None :
            start_langevin_step = 1

        # Init timers
        total_elapsed_time = {}
        total_elapsed_time["neural_net"] = 0.0
        total_elapsed_time["pseudo"] = 0.0
        total_elapsed_time["am"] = 0.0
        total_elapsed_time["energy"] = 0.0
        total_elapsed_time["random_noise"] = 0.0

        total_saddle_iter = 0
        total_error_level = 0.0
        total_net_failed = 0
        time_start = time.time()

        #------------------ run ----------------------
        print("iteration, mass error, total partitions, total energy, incompressibility error (or saddle point error)")
        print("---------- Run  ----------")
        for langevin_step in range(start_langevin_step, max_step+1):
            print("Langevin step: ", langevin_step)

            time_random_noise = time.time()

            # Update w_minus using Leimkuhler-Matthews method
            normal_noise_current = self.random.normal(0.0, self.langevin["sigma"], [R, self.cb.get_n_grid()])
            w_lambda = np.zeros([R, self.cb.get_n_grid()], dtype=np.float64) # array for output fields
            
            for count, i in enumerate(self.exchange_fields_real_idx):
                w_lambda[count] -= 1.0/self.exchange_eigenvalues[i]*w_exchange[i]
            for count, i in enumerate(self.exchange_fields_real_idx):
                for j in range(S-1):
                    w_lambda[count] += 1.0/self.exchange_eigenvalues[i]*self.matrix_o[j,i]*self.vector_s[j]
                    w_lambda[count] += self.matrix_o[j,i]*phi[self.monomer_types[j]]

            w_exchange[self.exchange_fields_real_idx] += -w_lambda*self.langevin["dt"] + (normal_noise_prev + normal_noise_current)/2

            # Swap two noise arrays
            normal_noise_prev, normal_noise_current = normal_noise_current, normal_noise_prev

            total_elapsed_time["random_noise"] += time.time() - time_random_noise

            # Find saddle point of the pressure field
            _, phi, saddle_iter, error_level, elapsed_time, is_net_failed = \
                self.find_saddle_point(w_exchange=w_exchange, tolerance=self.saddle["tolerance"], net=self.net)

            # Update timers
            total_elapsed_time["pseudo"] += elapsed_time["pseudo"]
            total_elapsed_time["neural_net"] += elapsed_time["neural_net"]
            total_elapsed_time["am"] += elapsed_time["am"]
            total_elapsed_time["energy"] += elapsed_time["energy"]
            
            total_saddle_iter += saddle_iter
            total_error_level += error_level
            if (is_net_failed): total_net_failed += 1
            if (np.isnan(error_level) or error_level >= self.saddle["tolerance"]):
                print("Could not satisfy tolerance")
                return total_saddle_iter/langevin_step, total_error_level/langevin_step

            # Calculate structure function
            if langevin_step % self.recording["sf_computing_period"] == 0:
                # Perform Fourier transforms
                mu_fourier = {}
                phi_fourier = {}
                for i in range(S):
                    key = self.monomer_types[i]
                    phi_fourier[key] = np.fft.rfftn(np.reshape(phi[self.monomer_types[i]], self.cb.get_nx()))/self.cb.get_n_grid()
                    mu_fourier[key] = np.zeros_like(np.fft.rfftn(np.reshape(w_exchange[0], self.cb.get_nx())), np.complex128)
                    for k in range(S-1) :
                        mu_fourier[key] += np.fft.rfftn(np.reshape(w_exchange[k], self.cb.get_nx()))*self.matrix_a_inv[k,i]/self.exchange_eigenvalues[k]/self.cb.get_n_grid()
                # Accumulate S_ij(K) 
                for monomer_id_pair in itertools.combinations_with_replacement(list(range(S)),2):
                    sorted_pair = sorted(monomer_id_pair)
                    i = sorted_pair[0]
                    j = sorted_pair[1]
                    type_pair = self.monomer_types[i] + "," + self.monomer_types[j]

                    # Assuming that <u(k)>*<phi(-k)> is zero in the disordered phase
                    sf_average[type_pair] += mu_fourier[self.monomer_types[i]]* np.conj( phi_fourier[self.monomer_types[j]])

            # Save structure function
            if langevin_step % self.recording["sf_recording_period"] == 0:
                for monomer_id_pair in itertools.combinations_with_replacement(list(range(S)),2):
                    sorted_pair = sorted(monomer_id_pair)
                    i = sorted_pair[0]
                    j = sorted_pair[1]
                    type_pair = self.monomer_types[i] + "," + self.monomer_types[j]
                    sf_average[type_pair] *= self.recording["sf_computing_period"]/self.recording["sf_recording_period"]* \
                            self.cb.get_volume()*np.sqrt(self.langevin["nbar"])

                # Make a dictionary for chi_n
                chi_n_mat = {}
                for pair_chi_n in self.params["chi_n"]:
                    sorted_name_pair = sorted(pair_chi_n[0:2])
                    chi_n_mat[sorted_name_pair[0] + "," + sorted_name_pair[1]] = pair_chi_n[2]

                mdic = {"dim":self.cb.get_dim(), "nx":self.cb.get_nx(), "lx":self.cb.get_lx(),
                    "chi_n":chi_n_mat, "chain_model":self.chain_model, "ds":self.ds,
                    "dt":self.langevin["dt"], "nbar":self.langevin["nbar"], "initial_params":self.params,
                    "structure_function":sf_average}
                savemat(os.path.join(self.recording["dir"], "structure_function_%06d.mat" % (langevin_step)), mdic)
                
                # Reset Arrays
                for monomer_id_pair in itertools.combinations_with_replacement(list(range(S)),2):
                    sorted_pair = sorted(monomer_id_pair)
                    i = sorted_pair[0]
                    j = sorted_pair[1]
                    type_pair = self.monomer_types[i] + "," + self.monomer_types[j]
                    sf_average[type_pair][:,:,:] = 0.0

            # Save simulation data
            if (langevin_step) % self.recording["recording_period"] == 0:
                w = np.matmul(self.matrix_a, w_exchange)
                self.save_simulation_data(
                    path=os.path.join(self.recording["dir"], "fields_%06d.mat" % (langevin_step)),
                    w=w, phi=phi, langevin_step=langevin_step, normal_noise_prev=normal_noise_prev)

        # estimate execution time
        time_duration = time.time() - time_start
        print("\nTotal elapsed time: %f, Elapsed time per Langevin step: %f" %
            (time_duration, time_duration/(max_step+1-start_langevin_step)))
        print("Total iterations for saddle points: %d, Iterations per Langevin step: %f" %
            (total_saddle_iter, total_saddle_iter/(max_step+1-start_langevin_step)))
        print("Elapsed time ratio:")
        print("\tPseudo: %f" % (total_elapsed_time["pseudo"]/time_duration))
        print("\tDeep learning : %f" % (total_elapsed_time["neural_net"]/time_duration))
        print("\tAnderson mixing: %f" % (total_elapsed_time["am"]/time_duration))
        print("\tHamiltonian and its derivative: %f" % (total_elapsed_time["energy"]/time_duration))
        print("\tRandom noise: %f" % (total_elapsed_time["random_noise"]/time_duration))
        print( "The number of times that the neural-net could not reduce the incompressibility error and switched to Anderson mixing: %d times" % 
            (total_net_failed))
        return total_saddle_iter/(max_step+1-start_langevin_step), total_error_level/(max_step+1-start_langevin_step)

    def find_saddle_point(self, w_exchange, tolerance, net=None):

        # The number of components
        S = len(self.monomer_types)

        # The numbers of real and imaginary fields respectively
        R = self.R
        I = self.I
        
        # Assign large initial value for the energy and error
        energy_total = 1e20
        error_level = 1e20

        # Reset Anderson mixing module
        self.am.reset_count()

        # Concentration of each monomer
        phi = {}

        # Compute hamiltonian part that is only related to real-valued fields
        energy_total_real = 0.0
        for count, i in enumerate(self.exchange_fields_real_idx):
            energy_total_real -= 0.5/self.exchange_eigenvalues[i]*np.dot(w_exchange[i], w_exchange[i])/self.cb.get_n_grid()
        for count, i in enumerate(self.exchange_fields_real_idx):
            for j in range(S-1):
                energy_total_real += 1.0/self.exchange_eigenvalues[i]*self.matrix_o[j,i]*self.vector_s[j]*np.mean(w_exchange[i])

        # Compute hamiltonian part that is constant
        for i in range(S-1):
            energy_ref = 0.0
            for j in range(S-1):
                energy_ref += self.matrix_o[j,i]*self.vector_s[j]
            energy_total_real -= 0.5*energy_ref**2/self.exchange_eigenvalues[i]

        # Init timers
        elapsed_time = {}
        elapsed_time["neural_net"] = 0.0
        elapsed_time["pseudo"] = 0.0
        elapsed_time["am"] = 0.0
        elapsed_time["energy"] = 0.0
        is_net_failed = False

        # Saddle point iteration begins here
        for saddle_iter in range(1,self.saddle["max_iter"]+1):

            # Convert to species chemical potential fields
            w = np.matmul(self.matrix_a, w_exchange)

            # Make a dictionary for input fields 
            w_input = {}
            for i in range(S):
                w_input[self.monomer_types[i]] = w[i]
            for random_polymer_name, random_fraction in self.random_fraction.items():
                w_input[random_polymer_name] = np.zeros(self.cb.get_n_grid(), dtype=np.float64)
                for monomer_type, fraction in random_fraction.items():
                    w_input[random_polymer_name] += w_input[monomer_type]*fraction

            # For the given fields, compute the polymer statistics
            time_p_start = time.time()
            self.pseudo.compute_statistics(w_input)
            elapsed_time["pseudo"] += time.time() - time_p_start

            time_e_start = time.time()
            # Compute total concentration for each monomer type
            phi = {}
            for monomer_type in self.monomer_types:
                phi[monomer_type] = self.pseudo.get_total_concentration(monomer_type)

            # Add random copolymer concentration to each monomer type
            for random_polymer_name, random_fraction in self.random_fraction.items():
                phi[random_polymer_name] = self.pseudo.get_total_concentration(random_polymer_name)
                for monomer_type, fraction in random_fraction.items():
                    phi[monomer_type] += phi[random_polymer_name]*fraction

            # Calculate incompressibility and saddle point error
            h_deriv = np.zeros([I, self.cb.get_n_grid()], dtype=np.float64)
            for count, i in enumerate(self.exchange_fields_imag_idx):
                if i != S-1:
                    h_deriv[count] -= 1.0/self.exchange_eigenvalues[i]*w_exchange[i]
            for count, i in enumerate(self.exchange_fields_imag_idx):
                if i != S-1:
                    for j in range(S-1):
                        h_deriv[count] += 1.0/self.exchange_eigenvalues[i]*self.matrix_o[j,i]*self.vector_s[j]
                        h_deriv[count] += self.matrix_o[j,i]*phi[self.monomer_types[j]]
            for i in range(S):
                h_deriv[I-1] += phi[self.monomer_types[i]]
            h_deriv[I-1] -= 1.0

            # Compute total error
            old_error_level = error_level
            error_level_list = []
            for i in range(I):
                error_level_list.append(np.std(h_deriv[i]))
            error_level = np.max(error_level_list)

            # Print iteration # and error levels
            if(self.verbose_level == 2 or self.verbose_level == 1 and
            (error_level < tolerance or saddle_iter == self.saddle["max_iter"])):

                # Calculate the total energy
                energy_total = energy_total_real - np.mean(w_exchange[S-1])
                for count, i in enumerate(self.exchange_fields_imag_idx):
                    if i != S-1:
                        energy_total -= 0.5/self.exchange_eigenvalues[i]*np.dot(w_exchange[i], w_exchange[i])/self.cb.get_n_grid()
                for count, i in enumerate(self.exchange_fields_imag_idx):
                    if i != S-1:
                        for j in range(S-1):
                            energy_total += 1.0/self.exchange_eigenvalues[i]*self.matrix_o[j,i]*self.vector_s[j]*np.mean(w_exchange[i])
                for p in range(self.mixture.get_n_polymers()):
                    energy_total -= self.mixture.get_polymer(p).get_volume_fraction()/ \
                                    self.mixture.get_polymer(p).get_alpha() * \
                                    np.log(self.pseudo.get_total_partition(p))

                # Check the mass conservation
                mass_error = np.mean(h_deriv[I-1])
                print("%8d %12.3E " % (saddle_iter, mass_error), end=" [ ")
                for p in range(self.mixture.get_n_polymers()):
                    print("%13.7E " % (self.pseudo.get_total_partition(p)), end=" ")
                print("] %15.9f   [" % (energy_total), end="")
                for i in range(I):
                    print("%13.7E" % (error_level_list[i]), end=" ")
                print("]")
            elapsed_time["energy"] += time.time() - time_e_start

            # When neural net fails
            if net and is_net_failed == False and (error_level >= old_error_level or np.isnan(error_level)):
                # Restore fields from backup
                w_exchange[self.exchange_fields_imag_idx] = w_imag_backup
                is_net_failed = True
                print("\tNeural-net could not reduce the incompressibility error (or saddle point error) and switched to Anderson mixing.")
                continue

            # Conditions to end the iteration
            if error_level < tolerance:
                break

            if net and not is_net_failed:
                time_d_start = time.time()
                # Make a backup of imaginary fields
                w_imag_backup = w_exchange[self.exchange_fields_imag_idx].copy()
                # Make an array of real fields
                w_real = w_exchange[self.exchange_fields_real_idx]
                # Predict field difference using neural network
                w_imag_diff = net.predict_w_imag(w_real, h_deriv, self.cb.get_nx())
                # Update fields
                w_exchange[self.exchange_fields_imag_idx] += w_imag_diff
                elapsed_time["neural_net"] += time.time() - time_d_start
            else:
                # Calculate new fields using simple and Anderson mixing
                time_a_start = time.time()
                w_exchange[self.exchange_fields_imag_idx] = \
                    np.reshape(self.am.calculate_new_fields(w_exchange[self.exchange_fields_imag_idx],
                    h_deriv, old_error_level, error_level), [I, self.cb.get_n_grid()])
                elapsed_time["am"] += time.time() - time_a_start

        # Set mean of pressure field to zero
        w_exchange[S-1] -= np.mean(w_exchange[S-1])

        return w_exchange[self.exchange_fields_imag_idx], \
            phi, saddle_iter, error_level, elapsed_time, is_net_failed
