import os
import pathlib
import numpy as np
from scipy.io import *
from langevinfts import *
from find_saddle_point import *
from deep_fts import *

# -------------- major parameters ------------

# Deep Learning
use_pretrained_model = True
pretrained_model_file = "pretrained_models/gyroid_atrpar_32.pth"

# Load Data
input_data = loadmat("eq_inputs/data_simulation_chin18.0.mat", squeeze_me=True)

# Simulation Box
nx = input_data['nx'].tolist() 
lx = input_data['lx'].tolist()

# Polymer Chain
n_contour = input_data['N']
f = input_data['f']
chi_n = input_data['chi_n']
chain_model = input_data['chain_model']

# Read initial fields
w_plus = input_data["w_plus"]
w_minus = input_data["w_minus"]

# Anderson Mixing
saddle_tolerance = 1e-4
saddle_max_iter = 100
am_n_comp = 1  # W+
am_max_hist= 20
am_start_error = 1e-1
am_mix_min = 0.1
am_mix_init = 0.1

# Langevin Dynamics
langevin_dt = input_data['langevin_dt']  # langevin step interval, delta tau*N
langevin_nbar = input_data['nbar']       # invariant polymerization index
langevin_recording_period = 1000
langevin_max_iter = 100

#------------- non-polymeric parameters -----------------------------

# OpenMP environment variables 
os.environ["MKL_NUM_THREADS"] = "1"  # always 1
os.environ["OMP_STACKSIZE"] = "1G"
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"  # 0, 1 or 2
torch.set_num_threads(1)

# Cuda environment variables 
os.environ["CUDA_VISIBLE_DEVICES"]= "1"

verbose_level = 1  # 1 : print at each langevin step.
                   # 2 : print at each saddle point iteration.

# random seed for MT19937
#np.random.seed(5489)

# -------------- initialize ------------
# choose platform among [cuda, cpu-mkl, cpu-fftw]
factory = PlatformSelector.create_factory("cuda")

# create polymer simulation instances
sb     = factory.create_simulation_box(nx, lx)
pc     = factory.create_polymer_chain(f, n_contour, chi_n, chain_model)
pseudo = factory.create_pseudo(sb, pc)
am     = factory.create_anderson_mixing(sb, am_n_comp,
          am_max_hist, am_start_error, am_mix_min, am_mix_init)

# create deep learning model
if (use_pretrained_model):
    net = DeepFts(dim=3, mid_channels=32)
    net.load_state_dict(torch.load(pretrained_model_file), strict=True)
else:
    net = None
    
# -------------- print simulation parameters ------------
print("---------- Simulation Parameters ----------");
print("Box Dimension: %d"  % (sb.get_dim()) )
print("Precision: 8")
print("chi_n: %f, f: %f, N: %d" % (pc.get_chi_n(), pc.get_f(), pc.get_n_contour()) )
print("%s chain model" % (pc.get_model_name()) )
print("Nx: %d, %d, %d" % (sb.get_nx(0), sb.get_nx(1), sb.get_nx(2)) )
print("Lx: %f, %f, %f" % (sb.get_lx(0), sb.get_lx(1), sb.get_lx(2)) )
print("dx: %f, %f, %f" % (sb.get_dx(0), sb.get_dx(1), sb.get_dx(2)) )
print("Volume: %f" % (sb.get_volume()) )
print("Invariant Polymerization Index: %d" % (langevin_nbar) )
print("Random Number Generator: ", np.random.RandomState().get_state()[0])

#-------------- allocate array ------------
# free end initial condition. q1 is q and q2 is qdagger.
# q1 starts from A end and q2 starts from B end.
q1_init = np.ones( sb.get_n_grid(), dtype=np.float64)
q2_init = np.ones( sb.get_n_grid(), dtype=np.float64)
phi_a   = np.zeros(sb.get_n_grid(), dtype=np.float64)
phi_b   = np.zeros(sb.get_n_grid(), dtype=np.float64)

# keep the level of field value
sb.zero_mean(w_plus);
sb.zero_mean(w_minus);

#------------------ run ----------------------
(total_saddle_iter, saddle_iter_per, time_duration_per,
time_pseudo_ratio, time_neural_net_ratio, total_net_failed) \
    = run_langevin_dynamics(
        sb=sb, pc=pc, pseudo=pseudo, am=am, 
        q1_init=q1_init, q2_init=q2_init,
        phi_a=phi_a, phi_b=phi_b, w_plus=w_plus, w_minus=w_minus,
        saddle_max_iter=saddle_max_iter, saddle_tolerance=saddle_tolerance,
        max_iter=langevin_max_iter, dt=langevin_dt, nbar=langevin_nbar,
        verbose_level=1, net=net)

# estimate execution time
print( "Total iterations for saddle points: %d, iter per step: %f" %
    (total_saddle_iter, saddle_iter_per))
print( "Total time: %f, time per step: %f" %
    (time_duration_per*total_saddle_iter, time_duration_per) )
print( "Pseudo time ratio: %f, deep learning time ratio: %f" %
    (time_pseudo_ratio, time_neural_net_ratio) )
print( "The number of times that the neural-net could not reduce the incompressible error and switched to Anderson mixing: %d times" % (total_net_failed) )
