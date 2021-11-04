import sys
import os
import time
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from langevinfts import *

def save_training_data(path, name, langevin_step, idx, w_minus, w_plus, g_plus, w_plus_diff):
    out_file_name = "%s_%06d_%03d.npz" % (name, langevin_step, idx)
    np.savez( os.path.join(path, out_file_name),
        nx=nx, lx=lx, N=NN, f=pc.get_f(), chi_n=pc.get_chi_n(),
        polymer_model=polymer_model, n_bar=langevin_nbar,
        w_minus=w_minus.astype(np.float32),
        w_plus=w_plus.astype(np.float32),
        g_plus=g_plus.astype(np.float32),
        w_plus_diff=w_plus_diff.astype(np.float32))

def find_saddle_point():
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

        # store training data 
        if (error_level > saddle_tolerance_for_simulation and
            saddle_iter % training_data_period ==0 ):
            training_data_list.append( {"w_plus":w_plus.copy(), "g_plus":g_plus.copy(), "iter":saddle_iter} )

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
            return training_data_list
            #break;
        
        # calculte new fields using simple and Anderson mixing
        w_plus_out = w_plus + g_plus 
        sb.zero_mean(w_plus_out);
        am.caculate_new_fields(w_plus, w_plus_out, g_plus, old_error_level, error_level);

# -------------- simulation parameters ------------
# OpenMP environment variables 
os.environ["KMP_STACKSIZE"] = "1G"
os.environ["MKL_NUM_THREADS"] = "1"  # always 1
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"  # 0, 1 or 2

data_path = "data1d_64_wp_diff/eval"
pathlib.Path(data_path).mkdir(parents=True, exist_ok=True)

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

# -------------- print simulation parameters ------------
print("---------- Simulation Parameters ----------");
print("Box Dimension: %d"  % (sb.get_dimension()) )
print("Precision: 8")
print("chi_n: %f, f: %f, NN: %d" % (pc.get_chi_n(), pc.get_f(), pc.get_NN()) )
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
q1_init = np.ones( sb.get_MM(), dtype=np.float64)
q2_init = np.ones( sb.get_MM(), dtype=np.float64)
phi_a   = np.zeros(sb.get_MM(), dtype=np.float64)
phi_b   = np.zeros(sb.get_MM(), dtype=np.float64)

print("wminus and wplus are initialized to random")
w_plus = np.random.normal(0, langevin_sigma, sb.get_MM())
w_minus = np.random.normal(0, langevin_sigma, sb.get_MM())

# keep the level of field value
sb.zero_mean(w_plus);
sb.zero_mean(w_minus);

find_saddle_point()
#------------------ run ----------------------
print("---------- Run ----------")
time_start = time.time()

#print("iteration, mass error, total_partition, energy_total, error_level")
for langevin_step in range(0, langevin_max_iter):
    
    print("langevin step: ", langevin_step)
    # update w_minus: predict step
    w_minus_copy = w_minus.copy()
    normal_noise = np.random.normal(0.0, langevin_sigma, sb.get_MM())
    lambda1 = phi_a-phi_b + 2*w_minus/pc.get_chi_n()
    w_minus += -lambda1*langevin_dt + normal_noise
    sb.zero_mean(w_minus)
    training_data_list = find_saddle_point()
    if( (langevin_step < 50000 and langevin_step % 4 == 0) or
        (langevin_step % 200 == 0) ):
        for data in training_data_list:
            save_training_data(data_path, "fields_1", langevin_step, data["iter"], 
            w_minus, data["w_plus"], data["g_plus"], w_plus-data["w_plus"])

    # update w_minus: correct step 
    lambda2 = phi_a-phi_b + 2*w_minus/pc.get_chi_n()
    w_minus = w_minus_copy - 0.5*(lambda1+lambda2)*langevin_dt + normal_noise
    sb.zero_mean(w_minus)
    training_data_list = find_saddle_point()
    if( (langevin_step < 50000 and langevin_step % 4 == 0) or
        (langevin_step % 200 == 0) ):
        for data in training_data_list:
            save_training_data(data_path, "fields_2", langevin_step, data["iter"], 
            w_minus, data["w_plus"], data["g_plus"], w_plus-data["w_plus"])

# estimate execution time
time_duration = time.time() - time_start; 
print( "total time: %f, time per step: %f" %
    (time_duration, time_duration/langevin_max_iter) )
