import sys
import os
import time
import pathlib
import numpy as np
from langevinfts import *
from find_saddle_point import *

# -------------- major parameters ------------

# Simulation Box
nx = [64, 64, 64]
lx = [7.31, 7.31, 7.31]

# Polymer Chain
n_contour = 90
f = 0.4
chi_n = 18.0
chain_model = "Discrete"

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
langevin_dt = 0.8       # langevin step interval, delta tau*N
langevin_nbar = 10000   # invariant polymerization index

#------------- non-polymeric parameters -----------------------------

os.environ["CUDA_VISIBLE_DEVICES"]= "0"

training_data_dir = "data_training_2"
pathlib.Path(training_data_dir).mkdir(parents=True, exist_ok=True)

verbose_level = 1  # 1 : print at each langevin step.
                   # 2 : print at each saddle point iteration.

# random seed for MT19937
#np.random.seed(5489)

langevin_max_iter_training = 10000
recording_period_train = 5
recording_n_data = 3

# -------------- initialize ------------
# choose platform among [cuda, cpu-mkl, cpu-fftw]
factory = PlatformSelector.create_factory("cuda")

# create polymer simulation instances
sb     = factory.create_simulation_box(nx, lx)
pc     = factory.create_polymer_chain(f, n_contour, chi_n, chain_model)
pseudo = factory.create_pseudo(sb, pc)
am     = factory.create_anderson_mixing(sb, am_n_comp,
          am_max_hist, am_start_error, am_mix_min, am_mix_init)

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

#print("wminus and wplus are initialized to random")
#w_plus = np.random.normal(0, langevin_sigma, sb.get_n_grid())
#w_minus = np.random.normal(0, langevin_sigma, sb.get_n_grid())

input_data = np.load("GyroidScftInput.npz")
w_plus = (input_data["w"][0] + input_data["w"][1])/2
w_minus = (input_data["w"][0] - input_data["w"][1])/2

# keep the level of field value
sb.zero_mean(w_plus);
sb.zero_mean(w_minus);

# make training data
make_training_data(
    sb=sb, pc=pc, pseudo=pseudo, am=am, 
    q1_init=q1_init, q2_init=q2_init,
    phi_a=phi_a, phi_b=phi_b, w_plus=w_plus, w_minus=w_minus,
    saddle_max_iter=saddle_max_iter, saddle_tolerance=saddle_tolerance, saddle_tolerance_ref=saddle_tolerance_ref,
    max_iter=langevin_max_iter_training, dt=langevin_dt, nbar=langevin_nbar,
    path_dir=training_data_dir,
    recording_period=recording_period_train,
    recording_n_data=recording_n_data, verbose_level=1)
