import sys
import os
import time
import pathlib
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from langevinfts import *
from train1d import *

# -------------- simulation parameters ------------
# OpenMP environment variables 
os.environ["KMP_STACKSIZE"] = "1G"
os.environ["MKL_NUM_THREADS"] = "1"  # always 1
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"  # 0, 1 or 2

#pp = ParamParser.get_instance()
#pp.read_param_file(sys.argv[1], False);
#pp.get("platform")
#os.environ["CUDA_VISIBLE_DEVICES"]= "0"
pathlib.Path("data").mkdir(parents=True, exist_ok=True)
folder_name = "/hdd/hdd2/yong/L_FTS/1d_periodic/data_64"
model_file = "S1_CP_epoch100.pth"
batch_size = 128

verbose_level = 1  # 1 : print at each langevin step.
                   # 2 : print at each saddle point iteration.

# Simulation Box
nx = [64]
lx = [9.6]

# Polymer Chain
NN = 80
f = 0.5
chi_n = 15
polymer_model = "Discrete"

# -------------- initialize ------------
# choose platform among [CUDA, CPU_MKL, CPU_FFTW]
factory = PlatformSelector.create_factory("CPU_MKL")

# create instances and assign to the variables of base classs
# for the dynamic binding
pc = factory.create_polymer_chain(f, NN, chi_n)
sb = factory.create_simulation_box(nx, lx)
pseudo = factory.create_pseudo(sb, pc, polymer_model)

# Deep Learning model FTS
model = DeepFts1d(model_file)

# -------------- print simulation parameters ------------
print("---------- Simulation Parameters ----------");
print("Box Dimension: %d"  % (sb.get_dimension()) )
print("Precision: 8")
print("chi_n: %f, f: %f, NN: %d" % (pc.get_chi_n(), pc.get_f(), pc.get_NN()) )
print("Nx: %d, %d, %d" % (sb.get_nx(0), sb.get_nx(1), sb.get_nx(2)) )
print("Lx: %f, %f, %f" % (sb.get_lx(0), sb.get_lx(1), sb.get_lx(2)) )
print("dx: %f, %f, %f" % (sb.get_dx(0), sb.get_dx(1), sb.get_dx(2)) )
print("Volume: %f" % (sb.get_volume()))

#-------------- allocate array ------------
q1_init = np.ones (sb.get_MM(), dtype=np.float64)
q2_init = np.ones (sb.get_MM(), dtype=np.float64)
phi_a   = np.zeros(sb.get_MM(), dtype=np.float64)
phi_b   = np.zeros(sb.get_MM(), dtype=np.float64)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f'Using device {device}')
logging.info(f'Current cuda device {torch.cuda.current_device()}')
logging.info(f'Count of using GPUs {torch.cuda.device_count()}')

net = AtrNet1d()
net.to(device=device)
net.load_state_dict(torch.load(model_file, map_location=device))
net.to(device=device)
logging.info(f'Model loaded from {model_file}')

val_data = FtsDataset1d(folder_name)
n_data = len(val_data)
net.eval()

file_list = glob.glob(folder_name + "/*.npz")
file_list.sort()
sample_data = np.load(file_list[0])
nx = sample_data["nx"]

with tqdm(total=n_data, desc='Validation', unit='batch', leave=False) as pbar:
    for file_name in file_list:
        # for the given fields find the polymer statistics
        data = np.load(file_name)
        w_minus = data["w_minus"]
        w_plus = data["w_plus"]
        w_plus_gen = model.generate_w_plus(np.reshape(w_minus, (1, 1, nx[0])), nx)  
        sb.zero_mean(w_plus_gen)
        QQ = pseudo.find_phi(phi_a, phi_b, q1_init,q2_init,
                    w_plus_gen + w_minus, w_plus_gen - w_minus)
        # calculate output fields
        g_plus = phi_a + phi_b - 1.0
        w_plus_diff = w_plus_gen - w_plus 
        """
        fig, axes = plt.subplots(2,2, figsize=(15,15))
        axes[0,0].plot(w_minus[:])
        axes[0,1].plot(w_plus[:])
        axes[0,1].plot(w_plus_gen[:])
        axes[1,0].plot(g_plus)
        axes[1,1].plot(w_plus_diff)
         
        plt.subplots_adjust(left=0.1,bottom=0.1,
                            top=0.9,right=0.9,
                            wspace=0.2, hspace=0.2)
        plt.savefig('w_plus_minus_%04d.png' % (i))
        """
        np.savez("data1D_64_diff/%s" % (os.path.basename(file_name)),
        nx=data["nx"], lx=data["lx"], N=data["N"], f=data["f"], chi_n=data["chi_n"],
        polymer_model=data["polymer_model"], n_bar=data["n_bar"], random_seed=data["random_seed"],
        w_minus=data["w_minus"], w_plus=data["w_plus"], phi_a=data["phi_a"], phi_b=data["phi_b"],
        w_plus_gen=w_plus_gen, g_plus=g_plus)
        
        pbar.update()
