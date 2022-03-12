import os
import pathlib
import numpy as np
from scipy.io import *
from langevinfts import *
from deep_fts import *
from find_saddle_point import *

# -------------- major parameters ------------

# Deep Learning
use_pretrained_model = True
pretrained_model_file = "pretrained_models/gyroid_asppnet.pth"

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
saddle_tolerance     = 1e-4
saddle_tolerance_ref = 1e-7
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
    net = DeepFts()
    net.load_state_dict(torch.load(pretrained_model_file), strict=True)
else:
    net = None

# standard deviation of normal noise for single segment
langevin_sigma = np.sqrt(2*langevin_dt*sb.get_n_grid()/ 
    (sb.get_volume()*np.sqrt(langevin_nbar)))
    
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
print("Langevin Sigma: %f" % (langevin_sigma) )
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

# find saddle point 
find_saddle_point(
    sb=sb, pc=pc, pseudo=pseudo, am=am,
    q1_init=q1_init, q2_init=q2_init,
    phi_a=phi_a, phi_b=phi_b,
    w_plus=w_plus, w_minus=w_minus,
    max_iter=saddle_max_iter,
    tolerance=saddle_tolerance,
    verbose_level=verbose_level)

# init timers
total_saddle_iter = 0
total_time_neural_net = 0.0
total_time_pseudo = 0.0
total_net_failed = 0
time_start = time.time()

#------------------ run ----------------------
print("iteration, mass error, total_partition, energy_total, error_level")
print("---------- Run  ----------")
for langevin_step in range(1, langevin_max_iter+1):
    
    print("Langevin step: ", langevin_step)
    # update w_minus: predict step
    w_minus_copy = w_minus.copy()
    normal_noise = np.random.normal(0.0, langevin_sigma, sb.get_n_grid())
    lambda1 = phi_a-phi_b + 2*w_minus/pc.get_chi_n()
    w_minus += -lambda1*langevin_dt + normal_noise
    sb.zero_mean(w_minus)
    (time_pseudo, time_neural_net, saddle_iter, error_level, is_net_failed) \
        = find_saddle_point(
            sb=sb, pc=pc, pseudo=pseudo, am=am, 
            q1_init=q1_init, q2_init=q2_init,
            phi_a=phi_a, phi_b=phi_b,
            w_plus=w_plus, w_minus=w_minus,
            max_iter=saddle_max_iter,
            tolerance=saddle_tolerance,
            verbose_level=verbose_level, net=net)
    total_time_pseudo += time_pseudo
    total_time_neural_net += time_neural_net
    total_saddle_iter += saddle_iter
    if (is_net_failed): total_net_failed += 1
    if (np.isnan(error_level) or error_level >= saddle_tolerance):
        print("Could not satisfy tolerance")
        break;

    # update w_minus: correct step 
    lambda2 = phi_a-phi_b + 2*w_minus/pc.get_chi_n()
    w_minus = w_minus_copy - 0.5*(lambda1+lambda2)*langevin_dt + normal_noise
    sb.zero_mean(w_minus)
    (time_pseudo, time_neural_net, saddle_iter, error_level, is_net_failed) \
        = find_saddle_point(
            sb=sb, pc=pc, pseudo=pseudo, am=am, 
            q1_init=q1_init, q2_init=q2_init,
            phi_a=phi_a, phi_b=phi_b,
            w_plus=w_plus, w_minus=w_minus,
            max_iter=saddle_max_iter,
            tolerance=saddle_tolerance,
            verbose_level=verbose_level, net=net)
    total_time_pseudo += time_pseudo
    total_time_neural_net += time_neural_net
    total_saddle_iter += saddle_iter
    if (is_net_failed): total_net_failed += 1
    if (np.isnan(error_level) or error_level >= saddle_tolerance):
        print("Could not satisfy tolerance")
        break;

# estimate execution time
print( "Total iterations for saddle points: %d, iter per step: %f" %
    (total_saddle_iter, total_saddle_iter/langevin_max_iter) )
time_duration = time.time() - time_start; 
print( "Total time: %f, time per step: %f" %
    (time_duration, time_duration/langevin_max_iter) )
print( "Pseudo time ratio: %f, deep learning time ratio: %f" %
    (total_time_pseudo/time_duration, total_time_neural_net/time_duration) )
print( "The number of times that the neural-net could not reduce the incompressible error and switched to Anderson mixing: %d times" % (total_net_failed) )
