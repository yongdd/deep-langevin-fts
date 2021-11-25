import sys
import os
import time
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from langevinfts import *

# -------------- simulation parameters ------------
# OpenMP environment variables 
os.environ["KMP_STACKSIZE"] = "1G"
os.environ["MKL_NUM_THREADS"] = "1"  # always 1
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"  # 0, 1 or 2

data_path = "data1d_64_wp_diff/eval"
pathlib.Path(data_path).mkdir(parents=True, exist_ok=True)

verbose_level = 1  # 1 : print at each langevin step.
                   # 2 : print at each saddle point iteration.

dim = 2
if (dim == 1):
    data_dir = "data1d_64_wp_diff"
    path = "data1d_64_wp_diff/val/"
elif (dim == 2):
    data_dir = "data2d_64_wp_diff"
    path = "data2d_64_wp_diff/val/"
elif (dim == 3):
    data_dir = "data3d_64_wp_diff"
    path = "data3d_64_wp_diff/val/"
     
if dim == 1 or dim == 2:
    langevin_iter = 400000 # 40000
elif dim == 3:
    langevin_iter = 40000
saddle_iter = 0
sample_file_name = path + "/fields_1_%06d_%03d.npz" % (langevin_iter, saddle_iter)
sample_data = np.load(sample_file_name)

# Simulation Box
nx = sample_data["nx"].tolist()
lx = sample_data["lx"].tolist()

# Polymer Chain
NN = 80
f = 0.5
chi_n = 15
polymer_model = "Discrete"

# Anderson Mixing 
saddle_tolerance_for_simulation = 1e-4
saddle_tolerance_for_training_data = 1e-7
saddle_max_iter = 500
am_n_comp = 1  # W+
am_max_hist= 20
am_start_error = 1e-1
am_mix_min = 0.1
am_mix_init = 0.1

# Langevin Dynamics
langevin_dt = 0.8     # langevin step interval, delta tau*N
langevin_nbar = 2000  # invariant polymerization index
langevin_max_iter = 500000

# -------------- initialize ------------
# choose platform among [CUDA, CPU_MKL, CPU_FFTW]
factory = PlatformSelector.create_factory("CPU_MKL")

# create instances and assign to the variables of base classs
# for the dynamic binding
pc = factory.create_polymer_chain(f, NN, chi_n)
sb = factory.create_simulation_box(nx, lx)
pseudo = factory.create_pseudo(sb, pc, polymer_model)
am = factory.create_anderson_mixing(sb, am_n_comp,
    am_max_hist, am_start_error, am_mix_min, am_mix_init)

# standard deviation of normal noise for single segment
langevin_sigma = np.sqrt(2*langevin_dt*sb.get_MM()/ 
    (sb.get_volume()*np.sqrt(langevin_nbar)))
    
# random seed for MT19937
np.random.seed(5489)

# training data is collected every 10 saddle point period
training_data_period = 5

#-------------- allocate array ------------
# free end initial condition. q1 is q and q2 is qdagger.
# q1 starts from A end and q2 starts from B end.
q1_init = np.ones( sb.get_MM(), dtype=np.float64)
q2_init = np.ones( sb.get_MM(), dtype=np.float64)
phi_a   = np.zeros(sb.get_MM(), dtype=np.float64)
phi_b   = np.zeros(sb.get_MM(), dtype=np.float64)

print("wminus and wplus are initialized to random")
w_plus = sample_data["w_plus"].astype(np.float64)
w_minus = sample_data["w_minus"].astype(np.float64)

# keep the level of field value
sb.zero_mean(w_plus);
sb.zero_mean(w_minus);

# find saddle point
# assign large initial value for the energy and error
energy_total = 1e20
error_level = 1e20

# reset Anderson mixing module
am.reset_count()

training_data_list = []
# saddle point iteration begins here
for saddle_iter in range(0,saddle_max_iter):

    # for the given fields find the polymer statistics
    QQ = pseudo.find_phi(phi_a, phi_b, 
            q1_init,q2_init,
            w_plus + w_minus,
            w_plus - w_minus)
    phi_plus = phi_a + phi_b

    # error_level measures the "relative distance" between the input and output fields
    old_error_level = error_level
    g_plus = phi_plus - 1.0
    error_level = np.sqrt(sb.inner_product(g_plus,g_plus)/sb.get_volume())


    # print iteration # and error levels
    if (verbose_level == 2 or
     verbose_level == 1 and
     (error_level < saddle_tolerance_for_training_data or saddle_iter == saddle_max_iter-1 )):
         
        # calculate the total energy
        energy_old = energy_total
        energy_total  = -np.log(QQ/sb.get_volume())
        energy_total += sb.inner_product(w_minus,w_minus)/pc.get_chi_n()/sb.get_volume()
        energy_total -= sb.integral(w_plus)/sb.get_volume()

        # check the mass conservation
        mass_error = sb.integral(phi_plus)/sb.get_volume() - 1.0
        print("%8d %12.3E %15.7E %13.9f %13.9f" %
            (saddle_iter, mass_error, QQ, energy_total, error_level))
    # conditions to end the iteration
    if(error_level < saddle_tolerance_for_training_data):
        break;

    # calculte new fields using simple and Anderson mixing
    w_plus_out = w_plus + g_plus 
    sb.zero_mean(w_plus_out);
    w_plus_copy = w_plus.copy()
    am.caculate_new_fields(w_plus, w_plus_out, g_plus, old_error_level, error_level);

    # store training data 
    if (error_level > saddle_tolerance_for_simulation and
        saddle_iter % training_data_period ==0 ):
        
        wm = w_minus
        wp = w_plus
        gp = g_plus
        
        sample_file_name = path + "/fields_1_%06d_%03d.npz" % (langevin_iter, saddle_iter)
        sample_data = np.load(sample_file_name)
        wpd = sample_data["w_plus_diff"]
        
        wpd_gen = w_plus - w_plus_copy
        X = np.linspace(0, lx[0], nx[0], endpoint=False)
        
        fig, axes = plt.subplots(2,2, figsize=(20,15))
        
        axes[0,0].plot(X, wm  [:nx[0]], )
        axes[0,1].plot(X, wp  [:nx[0]], )
        axes[1,0].plot(X, gp  [:nx[0]], )
        axes[1,1].plot(X, wpd [:nx[0]], )
        axes[1,1].plot(X, wpd_gen[:nx[0]], )

        #axes[1,0].set_ylim([-0.3, 0.4])
        #axes[1,1].set_ylim([-2.5, 2.5])
        #axes[1,0].set_ylim([-2.5, 2.5])
        #axes[1,1].set_ylim([-10, 10])

        plt.subplots_adjust(left=0.2,bottom=0.2,
                            top=0.8,right=0.8,
                            wspace=0.2, hspace=0.2)
        plt.savefig('w_plus_minus_%06d_%03d.png' % (langevin_iter, saddle_iter))
        
        print()
