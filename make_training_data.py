import sys
import os
import time
import pathlib
import numpy as np
from langevinfts import *
from find_saddle_point import *

def save_data(path, name, langevin_step, idx, w_minus, g_plus, w_plus_diff):
    out_file_name = "%s_%06d_%03d.npz" % (name, langevin_step, idx)
    np.savez( os.path.join(path, out_file_name),
        nx=nx, lx=lx, N=n_contour, f=pc.get_f(), chi_n=pc.get_chi_n(),
        polymer_model=chain_model, n_bar=langevin_nbar,
        w_minus=w_minus.astype(np.float16),
        g_plus=g_plus.astype(np.float16),
        w_plus_diff=w_plus_diff.astype(np.float16))

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

data_path_training = "data_training_2"
pathlib.Path(data_path_training  ).mkdir(parents=True, exist_ok=True)

verbose_level = 1  # 1 : print at each langevin step.
                   # 2 : print at each saddle point iteration.

# random seed for MT19937
#np.random.seed(5489)

langevin_max_iter_training = 10000
recording_period_train = 5
recording_n_random = 3

# -------------- initialize ------------
# choose platform among [cuda, cpu-mkl, cpu-fftw]
factory = PlatformSelector.create_factory("cuda")

# create polymer simulation instances
sb     = factory.create_simulation_box(nx, lx)
pc     = factory.create_polymer_chain(f, n_contour, chi_n, chain_model)
pseudo = factory.create_pseudo(sb, pc)
am     = factory.create_anderson_mixing(sb, am_n_comp,
          am_max_hist, am_start_error, am_mix_min, am_mix_init)

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

#print("wminus and wplus are initialized to random")
#w_plus = np.random.normal(0, langevin_sigma, sb.get_n_grid())
#w_minus = np.random.normal(0, langevin_sigma, sb.get_n_grid())

input_data = np.load("GyroidScftInput.npz")
w_plus = (input_data["w"][0] + input_data["w"][1])/2
w_minus = (input_data["w"][0] - input_data["w"][1])/2

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

#------------------ run ----------------------
print("iteration, mass error, total_partition, energy_total, error_level")
print("---------- Data Collection ----------")
for langevin_step in range(1, langevin_max_iter_training+1):
        print("Langevin step: ", langevin_step)
        # update w_minus
        normal_noise = np.random.normal(0.0, langevin_sigma, sb.get_n_grid())
        g_minus = phi_a-phi_b + 2*w_minus/pc.get_chi_n()
        w_minus += -g_minus*langevin_dt + normal_noise
        sb.zero_mean(w_minus)

        # find saddle point
        find_saddle_point(
            sb=sb, pc=pc, pseudo=pseudo, am=am, 
            q1_init=q1_init, q2_init=q2_init,
            phi_a=phi_a, phi_b=phi_b,
            w_plus=w_plus, w_minus=w_minus,
            max_iter=saddle_max_iter,
            tolerance=saddle_tolerance,
            verbose_level=verbose_level)
        w_plus_tol = w_plus.copy()
        g_plus_tol = phi_a + phi_b - 1.0

        # find more accurate saddle point
        find_saddle_point(
            sb=sb, pc=pc, pseudo=pseudo, am=am, 
            q1_init=q1_init, q2_init=q2_init,
            phi_a=phi_a, phi_b=phi_b,
            w_plus=w_plus, w_minus=w_minus,
            max_iter=saddle_max_iter,
            tolerance=saddle_tolerance_ref,
            verbose_level=verbose_level)
        w_plus_ref = w_plus.copy()

        # record data
        if (langevin_step % recording_period_train == 0):
            log_std_w_plus = np.log(np.std(w_plus))
            log_std_w_plus_diff = np.log(np.std(w_plus_tol - w_plus_ref))
            diff_exps = np.linspace(log_std_w_plus, log_std_w_plus_diff, num=recording_n_random+2)[1:-1]
            #print(diff_exps)
            for idx, exp in enumerate(diff_exps):
                std_w_plus_diff = np.exp(exp)
                #print(std_w_plus_diff)
                w_plus_noise = w_plus_ref + np.random.normal(0, std_w_plus_diff, sb.get_n_grid())
                QQ = pseudo.find_phi(
                        phi_a, phi_b,
                        q1_init, q2_init,
                        w_plus_noise + w_minus,
                        w_plus_noise - w_minus)
                g_plus = phi_a + phi_b - 1.0
                save_data(data_path_training, "fields_%d" % np.round(chi_n*100), langevin_step, idx, 
                w_minus, g_plus, w_plus_ref-w_plus_noise)
