import os
import time
import glob
import shutil
import pathlib
import numpy as np
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
    def __init__(self, dim, in_channels=3, mid_channels=32, out_channels=1, kernel_size = 3, epoch_offset=None):
        super().__init__()
        padding = (kernel_size-1)//2
        self.dim = dim
        self.loss = torch.nn.MSELoss()
        
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
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[100], gamma=0.2,
            verbose=False)
        return [optimizer], [scheduler]
    
    def on_train_start(self):
        total_params = sum(p.numel() for p in self.parameters())
        self.log('total_params', float(total_params))
        #print("total_params", total_params)
    
    def on_train_epoch_start(self):
        self.log('learning_rate', self.optimizers().param_groups[0]['lr'])
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
        self.log('train_loss', loss)
        return loss

    def predict_w_plus(self, d_w_minus, d_g_plus, nx):
    
        X = torch.zeros([1, 3, np.prod(nx)], dtype=torch.float64).to(self.device)
        X[0,0,:] = d_w_minus/10.0
        X[0,1,:] = d_g_plus
        
        # zero mean
        X[0,0,:] -= torch.mean(X[0,0,:])
        X[0,1,:] -= torch.mean(X[0,1,:])
        
        # normalization
        std_g_plus = torch.std(X[0,1,:])
        X[0,1,:] /= std_g_plus
        X[0,2,:] = torch.sqrt(std_g_plus)
        X = torch.reshape(X, [1, 3] + list(nx)).type(torch.float16)

        with torch.no_grad():
            output = self(X)*std_g_plus*20
            d_w_plus_diff = torch.reshape(output.type(torch.float64), (-1,))
            return d_w_plus_diff

class FtsDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir):

        file_list = sorted(glob.glob(data_dir + "/*.npz"))
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
        
        # zero mean
        X[0,:] -= np.mean(X[0,:])
        X[1,:] -= np.mean(X[1,:])
        Y[0,:] -= np.mean(Y[0,:])
        
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

def calculate_sigma(langevin_nbar, langevin_dt, n_grids, volume):
        return np.sqrt(2*langevin_dt*n_grids/(volume*np.sqrt(langevin_nbar)))

class DeepLangevinFTS:
    def __init__(self, params, random_seed=None):

        assert(len(params['segment_lengths']) == 2), \
            "Currently, only AB-type polymers are supported."
        assert(len(set(["A","B"]).intersection(set(params['segment_lengths'].keys())))==2), \
            "Use letters 'A' and 'B' for monomer types."
        assert(len(params["distinct_polymers"]) >= 1), \
            "There is no polymer chain."

        # (c++ class) Create a factory for given platform and chain_model
        if "reduce_gpu_memory_usage" in params:
            factory = PlatformSelector.create_factory("cuda", params["chain_model"], params["reduce_gpu_memory_usage"])
        else:
            factory = PlatformSelector.create_factory("cuda", params["chain_model"], False)
        factory.display_info()
        
        # (C++ class) Computation box
        cb = factory.create_computation_box(params["nx"], params["lx"])

        # Polymer chains
        total_volume_fraction = 0.0
        random_count = 0
        for polymer in params["distinct_polymers"]:
            block_length_list = []
            block_monomer_type_list = []
            v_list = []
            u_list = []
            A_fraction = 0.0
            alpha = 0.0  #total_relative_contour_length
            block_count = 0
            is_linear = not "v" in polymer["blocks"][0]
            for block in polymer["blocks"]:
                block_length_list.append(block["length"])
                block_monomer_type_list.append(block["type"])

                if is_linear:
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
                    
                alpha += block["length"]
                if block["type"] == "A":
                    A_fraction += block["length"]
                elif block["type"] == "random":
                    A_fraction += block["length"]*block["fraction"]["A"]
                block_count += 1

            total_volume_fraction += polymer["volume_fraction"]
            total_A_fraction = A_fraction/alpha
            statistical_segment_length = \
                np.sqrt(params["segment_lengths"]["A"]**2*total_A_fraction + \
                        params["segment_lengths"]["B"]**2*(1-total_A_fraction))

            if "random" in set(bt.lower() for bt in block_monomer_type_list):
                random_count +=1
                assert(random_count == 1), \
                    "Only one random copolymer is allowed." 
                assert(len(block_monomer_type_list) == 1), \
                    "Only single block random copolymer is allowed."
                assert(np.isclose(polymer["blocks"][0]["fraction"]["A"]+polymer["blocks"][0]["fraction"]["B"],1.0)), \
                    "The sum of volume fraction of random copolymer must be equal to 1."
                params["segment_lengths"].update({"R":statistical_segment_length})
                block_monomer_type_list = ["R"]
                self.random_copolymer_exist = True
                self.random_A_fraction = total_A_fraction

            else:
                self.random_copolymer_exist = False
            
            polymer.update({"block_monomer_types":block_monomer_type_list})
            polymer.update({"block_lengths":block_length_list})
            polymer.update({"v":v_list})
            polymer.update({"u":u_list})

        assert(np.isclose(total_volume_fraction,1.0)), "The sum of volume fraction must be equal to 1."

        # (C++ class) Mixture box
        if "use_superposition" in params:
            mixture = factory.create_mixture(params["ds"], params["segment_lengths"], params["use_superposition"])
        else:
            mixture = factory.create_mixture(params["ds"], params["segment_lengths"], True)

        # Add polymer chains
        for polymer in params["distinct_polymers"]:
            # print(polymer["volume_fraction"], polymer["block_monomer_types"], polymer["block_lengths"], polymer["v"], polymer["u"])
            mixture.add_polymer(polymer["volume_fraction"], polymer["block_monomer_types"], polymer["block_lengths"], polymer["v"] ,polymer["u"])

        # (C++ class) Solver using Pseudo-spectral method
        pseudo = factory.create_pseudo(cb, mixture)

        # (C++ class) Fields Relaxation using Anderson Mixing
        am = factory.create_anderson_mixing(
            np.prod(params["nx"]),      # the number of variables
            params["am"]["max_hist"],     # maximum number of history
            params["am"]["start_error"],  # when switch to AM from simple mixing
            params["am"]["mix_min"],      # minimum mixing rate of simple mixing
            params["am"]["mix_init"])     # initial mixing rate of simple mixing

        # Langevin Dynamics
        # standard deviation of normal noise
        langevin_sigma = calculate_sigma(params["langevin"]["nbar"], params["langevin"]["dt"],
            np.prod(params["nx"]), np.prod(params["lx"]))

        # set the number of threads for pytorch = 1
        torch.set_num_threads(1)

        # set torch device
        self.device = torch.device('cuda')

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
        
        print("%s chain model" % (params["chain_model"]))
        print("chi_n (N_ref): %f" % (params["chi_n"]))
        print("Conformational asymmetry (epsilon): %f" %
            (params["segment_lengths"]["A"]/params["segment_lengths"]["B"]))

        for p in range(mixture.get_n_polymers()):
            print("distinct_polymers[%d]:" % (p) )
            print("\tvolume fraction: %f, alpha: %f, N_total: %d" %
                (mixture.get_polymer(p).get_volume_fraction(),
                 mixture.get_polymer(p).get_alpha(),
                 mixture.get_polymer(p).get_n_segment_total()))
            # add display monomer types and lengths

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
        self.epsilon = params["segment_lengths"]["A"]/params["segment_lengths"]["B"]
        self.langevin = params["langevin"].copy()
        self.langevin.update({"sigma":langevin_sigma})
        self.langevin.pop("max_step", None)
        self.saddle = params["saddle"].copy()

        self.recording = params["recording"].copy()
        self.training = params["training"].copy()
        self.net = None

        self.verbose_level = params["verbose_level"]

        self.cb = cb
        self.mixture = mixture
        self.pseudo = pseudo
        self.am = am
        
    def save_training_data(self, path, w_minus, g_plus, w_plus_diff):
        np.savez(path,
            dim=self.cb.get_dim(), nx=self.cb.get_nx(), lx=self.cb.get_lx(),
            chi_n=self.chi_n, chain_model=self.chain_model, ds=self.ds, epsilon=self.epsilon,
            dt=self.langevin["dt"], nbar=self.langevin["nbar"], params=self.params,
            w_minus=w_minus.astype(np.float16),
            g_plus=g_plus.astype(np.float16),
            w_plus_diff=w_plus_diff.astype(np.float16))

        # mdic = {"dim":self.cb.get_dim(), "nx":self.cb.get_nx(), "lx":self.cb.get_lx(),
        #     "chi_n":self.chi_n, "chain_model":self.chain_model, "ds":self.ds, "epsilon":self.epsilon,
        #     "dt":self.langevin["dt"], "nbar":self.langevin["nbar"], "params":self.params,
        #     "w_minus":w_minus.astype(np.float16),
        #     "g_plus":g_plus.astype(np.float16),
        #     "w_plus_diff":w_plus_diff.astype(np.float16)}
        # savemat(path, mdic)

    def save_simulation_data(self, path, w_plus, w_minus, phi, langevin_step, normal_noise_prev):
        mdic = {"dim":self.cb.get_dim(), "nx":self.cb.get_nx(), "lx":self.cb.get_lx(),
            "chi_n":self.chi_n, "chain_model":self.chain_model, "ds":self.ds, "epsilon":self.epsilon,
            "dt":self.langevin["dt"], "nbar":self.langevin["nbar"], "params":self.params,
            "langevin_step":langevin_step,
            "random_generator":self.random_bg.state["bit_generator"],
            "random_state_state":str(self.random_bg.state["state"]["state"]),
            "random_state_inc":str(self.random_bg.state["state"]["inc"]),
            "normal_noise_prev":normal_noise_prev,
            "w_plus":w_plus, "w_minus":w_minus, "phi_a":phi["A"], "phi_b":phi["B"]}
        savemat(path, mdic)

    def make_training_data(self, w_plus, w_minus, last_training_step_file_name):

        # Skip make_training_data if this process is not the primary process of DPP in Pytorch-Lightning.
        # This is because DDP duplicates the main script when using multiple GPUs.
        is_secondary = os.environ.get("IS_DDP_SECONDARY")
        if is_secondary == "YES":
            return
        else:
            os.environ["IS_DDP_SECONDARY"] = "YES"

        # training data directory
        pathlib.Path(self.training["data_dir"]).mkdir(parents=True, exist_ok=True)

        # flattening arrays
        w_plus  = np.reshape(w_plus,  self.cb.get_n_grid())
        w_minus = np.reshape(w_minus, self.cb.get_n_grid())

        # create an empty array for field update algorithm
        d_normal_noise_prev = torch.zeros(self.cb.get_n_grid(), dtype=torch.float64).to(self.device)

        # find saddle point 
        w_plus, phi, _, _, _, _, _, _ = self.find_saddle_point(
            w_plus=w_plus, w_minus=w_minus, tolerance=self.saddle["tolerance"], net=None)

        #------------------ run ----------------------
        print("---------- Collect Training Data ----------")
        print("iteration, mass error, total_partition, energy_total, error_level")
        for langevin_step in range(1, self.training["max_step"]+1):
            print("Langevin step: ", langevin_step)

            # copy w_minus to gpu memory
            d_w_minus = torch.tensor(w_minus, dtype=torch.float64).to(self.device)

            # update w_minus using Leimkuhler-Matthews method
            d_normal_noise_current = torch.tensor(self.random.normal(0.0, self.langevin["sigma"], self.cb.get_n_grid()), dtype=torch.float64).to(self.device)
            d_lambda_minus = torch.tensor(phi["A"]-phi["B"], dtype=torch.float64).to(self.device) + 2*d_w_minus/self.chi_n
            d_w_minus += -d_lambda_minus*self.langevin["dt"] + (d_normal_noise_prev + d_normal_noise_current)/2

            # copy w_minus to host memory
            w_minus = d_w_minus.detach().cpu().numpy()

            # swap two noise arrays
            d_normal_noise_prev, d_normal_noise_current = d_normal_noise_current, d_normal_noise_prev

            # find saddle point
            w_plus_start = w_plus.copy()
            w_plus, phi, _, _, _, _, _, _ = self.find_saddle_point(
                w_plus=w_plus, w_minus=w_minus, tolerance=self.saddle["tolerance"], net=None)

            # training data is sampled from random noise distribution
            # with various standard deviations
            if (langevin_step % self.training["recording_period"] == 0):
                w_plus_tol = w_plus.copy()
                
                # find more accurate saddle point
                w_plus, phi, _, _, _, _, _, _ = self.find_saddle_point(
                    w_plus=w_plus, w_minus=w_minus, tolerance=self.training["tolerance"], net=None)
                w_plus_ref = w_plus.copy()
                phi_ref = phi.copy()

                sigma_a = np.std(w_plus_start - w_plus_ref)
                sigma_b = np.std(w_plus_tol   - w_plus_ref)
                
                log_sigma_sample = np.random.uniform(np.log(sigma_b), np.log(sigma_a), self.training["recording_n_data"])
                sigma_list = np.exp(log_sigma_sample)

                print(sigma_list)
                #print(np.log(sigma_list))
                for std_idx, sigma in enumerate(sigma_list):
                    w_plus_noise = w_plus_ref + np.random.normal(0, sigma, self.cb.get_n_grid())

                    # find g_plus for given distorted fields
                    if self.random_copolymer_exist:
                        self.pseudo.compute_statistics({"A":w_plus_noise+w_minus,"B":w_plus_noise-w_minus,"R":w_minus*(2*self.random_A_fraction-1)+w_plus_noise})
                    else:
                        self.pseudo.compute_statistics({"A":w_plus_noise+w_minus,"B":w_plus_noise-w_minus})

                    phi["A"] = self.pseudo.get_monomer_concentration("A")
                    phi["B"] = self.pseudo.get_monomer_concentration("B")

                    if self.random_copolymer_exist:
                        phi["R"] = self.pseudo.get_monomer_concentration("R")
                        phi["A"] += phi["R"]*self.random_A_fraction
                        phi["B"] += phi["R"]*(1.0-self.random_A_fraction)

                    g_plus = phi["A"] + phi["B"] - 1.0

                    path = os.path.join(self.training["data_dir"], "%d_%06d_%03d.npz" % (np.round(self.chi_n*100), langevin_step, std_idx))
                    print(path)
                    self.save_training_data(path, w_minus, g_plus, w_plus_ref-w_plus_noise)
            
        # save final configuration to use it as input in actual simulation
        self.save_simulation_data(path=last_training_step_file_name, 
            w_plus=w_plus_ref, w_minus=w_minus, phi=phi_ref,
            langevin_step=0,
            normal_noise_prev=d_normal_noise_prev.detach().cpu().numpy())

    def train_model(self, model_file=None, epoch_offset=None):
        torch.set_num_threads(1)

        print("---------- Training Parameters ----------")
        data_dir = self.training["data_dir"]
        batch_size = self.training["batch_size"]
        num_workers = self.training["num_workers"]
        gpus = self.training["gpus"] 
        num_nodes = self.training["num_nodes"] 
        max_epochs = self.training["max_epochs"] 
        precision = self.training["precision"]
        features = self.training["features"]

        print(f"data_dir: {data_dir}, batch_size: {batch_size}, num_workers: {num_workers}")
        print(f"gpus: {gpus}, num_nodes: {num_nodes}, max_epochs: {max_epochs}, precision: {precision}")

        print(f"---------- model file : {model_file} ----------")
        assert((model_file == None and epoch_offset == None) or
             (model_file != None and epoch_offset != None)), \
            "To continue the training, put both model file name and epoch offset."

        if model_file:
            self.net = TrainAndInference(dim=self.cb.get_dim(), mid_channels=features, epoch_offset=epoch_offset+1)
            self.net.load_state_dict(torch.load(model_file), strict=True)
            max_epochs -= epoch_offset+1
        else:
            self.net = TrainAndInference(dim=self.cb.get_dim(), mid_channels=features)
            
        # training data    
        train_dataset = FtsDataset(data_dir)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
        print(len(train_dataset))
        
        # training
        trainer = pl.Trainer(accelerator="gpu", devices=gpus,
                num_nodes=num_nodes, max_epochs=max_epochs, precision=precision,
                strategy=DDPStrategy(process_group_backend="gloo", find_unused_parameters=False),
                # process_group_backend="nccl" or "gloo"
                benchmark=True, log_every_n_steps=5)
        trainer.fit(self.net, train_loader, None)

    def find_best_epoch(self, w_plus, w_minus, best_epoch_file_name):

        # -------------- deep learning --------------
        saved_weight_dir = self.training["model_dir"]
        torch.set_num_threads(1)
        self.net = TrainAndInference(dim=self.cb.get_dim(), mid_channels=self.training["features"])
        self.net.set_inference_mode(self.device)

        #-------------- test roughly ------------
        list_saddle_iter_per = []
        file_list = sorted(glob.glob(saved_weight_dir + "/*.pth"), key=lambda l: (len(l), l))
        print(file_list)
        print("iteration, mass error, total_partition, energy_total, error_level")
        for model_file in file_list:
            saddle_iter_per, total_error_iter_per = self.run(
                w_plus=w_plus.copy(), w_minus=w_minus.copy(), max_step=10, model_file=model_file)
            if not np.isnan(total_error_iter_per):
                list_saddle_iter_per.append([model_file, saddle_iter_per])
        sorted_saddle_iter_per = sorted(list_saddle_iter_per, key=lambda l:l[1])

        #-------------- test top-10 epochs ------------
        list_saddle_iter_per = []
        for data in sorted_saddle_iter_per[0:10]:
            model_file = data[0]
            saddle_iter_per, total_error_iter_per = self.run(
                w_plus=w_plus.copy(), w_minus=w_minus.copy(), max_step=100, model_file=model_file)
            if not np.isnan(total_error_iter_per):
                list_saddle_iter_per.append([model_file, saddle_iter_per, total_error_iter_per])

        sorted_saddle_iter_per = sorted(list_saddle_iter_per, key=lambda l:(l[1], l[2]))
        print("\n\tfile name:    # iterations per langevin step,    total error per langevin step")
        for saddle_iter in sorted_saddle_iter_per:
            print("'%s': %5.2f, %12.3E" % tuple(saddle_iter), end = "\n")
        shutil.copy2(sorted_saddle_iter_per[0][0], best_epoch_file_name)
        print(f"\n'{sorted_saddle_iter_per[0][0]}' has been copied as '{best_epoch_file_name}'")


    def continue_simulation(self, file_name, max_step, model_file=None):

        # load_data
        load_data = loadmat(file_name, squeeze_me=True)

        # restore random state
        self.random_bg.state ={'bit_generator': 'PCG64',
            'state': {'state': int(load_data["random_state_state"]),
                      'inc':   int(load_data["random_state_inc"])},
                      'has_uint32': 0, 'uinteger': 0}
        print("Restored Random Number Generator: ", self.random_bg.state)

        # run
        self.run(w_plus=load_data["w_plus"], w_minus=load_data["w_minus"],
            max_step=max_step, model_file=model_file,
            normal_noise_prev=load_data["normal_noise_prev"],
            start_langevin_step=load_data["langevin_step"]+1)

    def run(self, w_plus, w_minus, max_step, model_file=None, normal_noise_prev=None, start_langevin_step=None):

        # simulation data directory
        pathlib.Path(self.recording["dir"]).mkdir(parents=True, exist_ok=True)

        # flattening arrays
        w_plus  = np.reshape(w_plus,  self.cb.get_n_grid())
        w_minus = np.reshape(w_minus, self.cb.get_n_grid())

        # structure function
        d_temp = torch.ones(self.cb.get_nx(), dtype=torch.float64).to(self.device)
        d_sf_average = torch.zeros_like(torch.fft.rfftn(d_temp), dtype=torch.float64).to(self.device)

        # load deep learning model weights
        print(f"---------- model file : {model_file} ----------")
        if model_file :
            self.net = TrainAndInference(dim=self.cb.get_dim(), mid_channels=self.training["features"])
            self.net.load_state_dict(torch.load(model_file), strict=True)
            self.net.set_inference_mode(self.device)

        # find saddle point 
        w_plus, phi, _, _, _, _, _, _ = self.find_saddle_point(w_plus=w_plus, w_minus=w_minus,
            tolerance=self.saddle["tolerance"], net=None)

        # create an empty array for field update algorithm
        if type(normal_noise_prev) == type(None) :
            d_normal_noise_prev = torch.zeros(self.cb.get_n_grid(), dtype=torch.float64).to(self.device)
        else:
            d_normal_noise_prev = torch.tensor(normal_noise_prev, dtype=torch.float64).to(self.device)

        if start_langevin_step == None :
            start_langevin_step = 1

        # init timers
        total_saddle_iter = 0
        total_error_level = 0.0
        total_time_neural_net = 0.0
        total_time_pseudo = 0.0
        total_time_am = 0.0
        total_time_random_noise = 0.0
        total_net_failed = 0
        time_start = time.time()

        #------------------ run ----------------------
        print("iteration, mass error, total partitions, total energy, incompressibility error")
        print("---------- Run  ----------")
        for langevin_step in range(start_langevin_step, max_step+1):
            print("Langevin step: ", langevin_step)

            time_random_noise = time.time() 
            # copy w_minus to gpu memory
            d_w_minus = torch.tensor(w_minus, dtype=torch.float64).to(self.device)

            # update w_minus using Leimkuhler-Matthews method
            d_normal_noise_current = torch.tensor(self.random.normal(0.0, self.langevin["sigma"], self.cb.get_n_grid()), dtype=torch.float64).to(self.device)
            d_lambda_minus = torch.tensor(phi["A"]-phi["B"], dtype=torch.float64).to(self.device) + 2*d_w_minus/self.chi_n
            d_w_minus += -d_lambda_minus*self.langevin["dt"] + (d_normal_noise_prev + d_normal_noise_current)/2
            
            # copy w_minus to host memory
            w_minus = d_w_minus.detach().cpu().numpy()

            # swap two noise arrays
            d_normal_noise_prev, d_normal_noise_current = d_normal_noise_current, d_normal_noise_prev

            total_time_random_noise += time.time() - time_random_noise

            # find saddle point of the pressure field
            w_plus, phi, saddle_iter, error_level, time_pseudo, time_neural_net, time_am, is_net_failed = \
                self.find_saddle_point(w_plus=w_plus, w_minus=w_minus,
                    tolerance=self.saddle["tolerance"], net=self.net)
            total_time_pseudo += time_pseudo
            total_time_neural_net += time_neural_net
            total_time_am += time_am
            total_saddle_iter += saddle_iter
            total_error_level += error_level
            if (is_net_failed): total_net_failed += 1
            if (np.isnan(error_level) or error_level >= self.saddle["tolerance"]):
                print("Could not satisfy tolerance")
                return total_saddle_iter/langevin_step, total_error_level/langevin_step

            # calculate structure function
            if langevin_step % self.recording["sf_computing_period"] == 0:
                d_sf_average += torch.absolute(torch.fft.rfftn(torch.reshape(d_w_minus, self.cb.get_nx()))/self.cb.get_n_grid())**2

            # save structure function
            if langevin_step % self.recording["sf_recording_period"] == 0:
                d_sf_average *= self.recording["sf_computing_period"]/self.recording["sf_recording_period"]* \
                        self.cb.get_volume()*np.sqrt(self.langevin["nbar"])/self.chi_n**2
                d_sf_average -= 1.0/(2*self.chi_n)
                mdic = {"dim":self.cb.get_dim(), "nx":self.cb.get_nx(), "lx":self.cb.get_lx(), "params": self.params,
                    "chi_n":self.chi_n, "chain_model":self.chain_model, "ds":self.ds, "epsilon":self.epsilon,
                    "dt": self.langevin["dt"], "nbar":self.langevin["nbar"], "structure_function":d_sf_average.detach().cpu().numpy()}
                savemat(os.path.join(self.recording["dir"], "structure_function_%06d.mat" % (langevin_step)), mdic)
                d_sf_average[:,:,:] = 0.0

            # save simulation data
            if (langevin_step) % self.recording["recording_period"] == 0:
                self.save_simulation_data(
                    path=os.path.join(self.recording["dir"], "fields_%06d.mat" % (langevin_step)),
                    w_plus=w_plus, w_minus=w_minus, phi=phi,
                    langevin_step=langevin_step,
                    normal_noise_prev=d_normal_noise_prev.detach().cpu().numpy())

        # estimate execution time
        time_duration = time.time() - time_start
        print( "Total iterations for saddle points: %d, iter per step: %f" %
            (total_saddle_iter, total_saddle_iter/max_step))
        print( "Total time: %f, time per step: %f" %
            (time_duration, time_duration/max_step))
        print("Pseudo time ratio: %f," % (total_time_pseudo/time_duration),
              "deep learning time ratio: %f," % (total_time_neural_net/time_duration),
              "Anderson mixing time ratio: %f," % (total_time_am/time_duration), 
              "Random noise time ratio: %f" % (total_time_random_noise/time_duration))
        print( "The number of times that the neural-net could not reduce the incompressible error and switched to Anderson mixing: %d times" % 
            (total_net_failed))
        return total_saddle_iter/max_step, total_error_level/max_step

    def find_saddle_point(self, w_plus, w_minus, tolerance, net=None):

        # assign large initial value for the energy and error
        energy_total = 1e20
        error_level = 1e20

        # reset Anderson mixing module
        self.am.reset_count()

        # concentration of each monomer
        phi = {}

        # compute hamiltonian part that is independent of w_plus
        d_w_minus = torch.tensor(w_minus, dtype=torch.float64).to(self.device)
        energy_total_minus = torch.dot(d_w_minus,d_w_minus).item()/self.chi_n/self.cb.get_n_grid()
        energy_total_minus += self.chi_n/4

        # copy w_plus to gpu memory
        d_w_plus = torch.tensor(w_plus, dtype=torch.float64).to(self.device)

        # tensor array for each monomer
        d_w = {}

        # init timers
        time_neural_net = 0.0
        time_pseudo = 0.0
        time_am = 0.0
        is_net_failed = False

        # saddle point iteration begins here
        for saddle_iter in range(1,self.saddle["max_iter"]+1):

            # make w_A, w_B, and w_R
            d_w["A"] = d_w_plus + d_w_minus
            d_w["B"] = d_w_plus - d_w_minus
            input_fields = {"A":d_w["A"].data_ptr(),"B":d_w["B"].data_ptr()}
            if self.random_copolymer_exist:
                d_w["R"] = d_w_minus*(2*self.random_A_fraction-1)+d_w_plus
                input_fields["R"] = d_w["R"].data_ptr()

            # for the given fields find the polymer statistics
            time_p_start = time.time()
            self.pseudo.compute_statistics_device(input_fields)
            time_pseudo += time.time() - time_p_start

            # get polymer concentration
            phi["A"] = self.pseudo.get_monomer_concentration("A")
            phi["B"] = self.pseudo.get_monomer_concentration("B")
            if self.random_copolymer_exist:
                phi["R"] = self.pseudo.get_monomer_concentration("R")
                phi["A"] += phi["R"]*self.random_A_fraction
                phi["B"] += phi["R"]*(1.0-self.random_A_fraction)

            # calculate output fields
            g_plus = phi["A"] + phi["B"] - 1.0
            d_g_plus = torch.tensor(g_plus, dtype=torch.float64).to(self.device)

            # error_level measures the "relative distance" between the input and output fields
            old_error_level = error_level
            error_level = torch.sqrt(torch.dot(d_g_plus,d_g_plus)/self.cb.get_n_grid()).item()

            # print iteration # and error levels
            if(self.verbose_level == 2 or self.verbose_level == 1 and
            (error_level < tolerance or saddle_iter == self.saddle["max_iter"])):

                # calculate the total energy
                energy_total = energy_total_minus - torch.mean(d_w_plus).item()
                for p in range(self.mixture.get_n_polymers()):
                    energy_total -= self.mixture.get_polymer(p).get_volume_fraction()/ \
                                    self.mixture.get_polymer(p).get_alpha() * \
                                    np.log(self.pseudo.get_total_partition(p))

                # check the mass conservation
                mass_error = torch.mean(d_g_plus).item()
                print("%8d %12.3E " % (saddle_iter, mass_error), end=" [ ")
                for p in range(self.mixture.get_n_polymers()):
                    print("%13.7E " % (self.pseudo.get_total_partition(p)), end=" ")
                print("] %15.9f %15.7E " % (energy_total, error_level))

            # when neural net fails
            if net and is_net_failed == False and (error_level >= old_error_level or np.isnan(error_level)):
                d_w_plus = d_w_plus_backup
                is_net_failed = True
                print("\tNeural-net could not reduce the incompressible error and switched to Anderson mixing.")
                continue

            # conditions to end the iteration
            if error_level < tolerance:
                break

            if net and not is_net_failed:
                # calculate new fields using neural network
                d_w_plus_backup = d_w_plus.detach().clone()
                time_d_start = time.time()
                d_w_plus_diff = net.predict_w_plus(d_w_minus, d_g_plus, self.cb.get_nx())
                torch.cuda.synchronize()
                d_w_plus += d_w_plus_diff
                time_neural_net += time.time() - time_d_start
            else:
                # calculate new fields using simple and Anderson mixing
                time_a_start = time.time()
                w_plus = d_w_plus.detach().cpu().numpy()
                w_plus = self.am.calculate_new_fields(w_plus, g_plus, old_error_level, error_level)
                d_w_plus = torch.tensor(w_plus, dtype=torch.float64).to(self.device)
                time_am += time.time() - time_a_start

        # make the mean zero
        d_w_plus -= torch.mean(d_w_plus)
        w_plus = d_w_plus.detach().cpu().numpy()

        return w_plus, phi, saddle_iter, error_level, time_pseudo, time_neural_net, time_am, is_net_failed
