import sys
import os
import time
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from langevinfts import *
from trainer_and_model import *
from deep_fts import *

def find_saddle_point(saddle_tolerance, use_net=False, plot=False):
    # assign large initial value for the energy and error
    energy_total = 1e20
    error_level = 1e20

    # reset Anderson mixing module
    am.reset_count()

    global phi_a
    global phi_b
    global w_plus
    global w_plus_acc
    global w_minus
    global time_dl
    global time_pseudo
    global total_saddle_iter
    model_loss = TrainerAndModel(dim=3)
    
    # saddle point iteration begins here
    for saddle_iter in range(0,saddle_max_iter):
        
        # for the given fields find the polymer statistics
        time_p_start = time.time()
        QQ = pseudo.find_phi(phi_a, phi_b, 
                q1_init,q2_init,
                w_plus + w_minus,
                w_plus - w_minus)
        time_pseudo += time.time() - time_p_start
        phi_plus = phi_a + phi_b
        
        # calculate output fields
        g_plus = phi_plus-1.0

        # error_level measures the "relative distance" between the input and output fields
        old_error_level = error_level
        error_level = np.sqrt(sb.inner_product(g_plus,g_plus)/sb.get_volume())

        # print iteration # and error levels
        if(verbose_level == 2 or
         verbose_level == 1 and
         (error_level < saddle_tolerance or saddle_iter == saddle_max_iter-1 )):
             
            # calculate the total energy
            energy_old = energy_total
            energy_total  = -np.log(QQ/sb.get_volume())
            energy_total += sb.inner_product(w_minus,w_minus)/pc.get_chi_n()/sb.get_volume()
            energy_total -= sb.integral(w_plus)/sb.get_volume()

            # check the mass conservation
            mass_error = sb.integral(phi_plus)/sb.get_volume() - 1.0
            print("%8d %12.3E %15.7E %13.9f %13.9f" %
                (saddle_iter+1, mass_error, QQ, energy_total, error_level))
        # conditions to end the iteration
        if(error_level < saddle_tolerance):
            total_saddle_iter += saddle_iter+1  
            break;
        
        wpd = w_plus_acc - w_plus
        sb.zero_mean(wpd)

        if (saddle_iter == 20):
            mdic = {"dim":sb.get_dim(), "nx":sb.get_nx(), "lx":sb.get_lx(),
                "N":pc.get_n_contour(), "f":pc.get_f(), "chi_n":pc.get_chi_n(),
                "chain_model":chain_model,
                "langevin_dt":langevin_dt, "nbar":langevin_nbar,
                "w_plus":w_plus, "w_minus":w_minus, "phi_a":phi_a, "phi_b":phi_b}
            sio.savemat("temp.mat", mdic)

        if (use_net):
            # predict new field using neural network
            time_d_start = time.time()
            w_plus_diff = deepfts.generate_w_plus(w_minus, g_plus, sb.get_nx()[:sb.get_dim()])
            w_plus += w_plus_diff
            sb.zero_mean(w_plus)
            time_dl += time.time() - time_d_start
            
            wpd_gen = w_plus_diff.copy()
            sb.zero_mean(wpd_gen)
            
            #print("loss", model_loss.NRMSLoss(torch.tensor(wpd), torch.tensor(wpd_gen)))
            #print("loss", model_loss.NRMSLoss(torch.tensor(wpd_gen), torch.tensor(wpd)))
            target = torch.tensor(wpd)
            output = torch.tensor(wpd_gen)
            
            print("mean01", torch.mean(target))
            print("mean02", torch.mean(output))
            
            print("mean1", torch.mean((target - output)**2))
            print("mean2", torch.mean(target**2))
            print("loss", torch.sqrt(torch.mean((target - output)**2))/torch.sqrt(torch.sqrt(torch.mean(target**2))))
            
        else:
            # calculte new fields using simple and Anderson mixing
            w_plus_out = w_plus + g_plus 
            sb.zero_mean(w_plus_out)
            w_plus_b_am = w_plus.copy()
            am.caculate_new_fields(w_plus, w_plus_out, g_plus, old_error_level, error_level);
            wpd_gen = w_plus.copy() - w_plus_b_am

        if plot == True:
            X = np.linspace(0, lx[0], nx[0], endpoint=False)
            wm = w_minus
            wp = w_plus
            gp = g_plus
            
            sb.zero_mean(wpd)
            sb.zero_mean(wpd_gen)
                               
            print(np.mean(wpd), np.mean(wpd_gen))
            fig, axes = plt.subplots(2,2, figsize=(20,15))
            
            #axes[0,0].plot(X, gp[:nx[0]])
            #axes[0,1].plot(X, wp[:nx[0]])
            #axes[1,0].plot(X, wpd[:nx[0]]/np.std(gp)/10)
            #axes[1,0].plot(X, wpd_gen[:nx[0]]/np.std(gp)/10)
            #axes[1,1].plot(X, wpd[:nx[0]])
            #axes[1,1].plot(X, wpd_gen[:nx[0]])

            plot_x1 = nx[0]*1245
            plot_x2 = nx[0]*1246
            axes[0,0].plot(X, wm     [plot_x1:plot_x2], )
            axes[0,1].plot(X, wp     [plot_x1:plot_x2], )
            axes[1,0].plot(X, gp     [plot_x1:plot_x2], )
            axes[1,1].plot(X, wpd    [plot_x1:plot_x2], )
            axes[1,1].plot(X, wpd_gen[plot_x1:plot_x2], )
            
            plt.ylim([-0.15, 0.15])
            plt.subplots_adjust(left=0.2,bottom=0.2,
                                top=0.8,right=0.8,
                                wspace=0.2, hspace=0.2)
            if (use_net):
                plt.savefig('use_net_y_%03d.png' % (saddle_iter))
            else:
                plt.savefig('use_net_n_%03d.png' % (saddle_iter))
            plt.close()

# -------------- simulation parameters ------------
# OpenMP environment variables 
os.environ["KMP_STACKSIZE"] = "1G"
os.environ["MKL_NUM_THREADS"] = "1"  # always 1
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "0"  # 0, 1 or 2
# Cuda environment variables 
os.environ["CUDA_VISIBLE_DEVICES"]= "0"

pathlib.Path("data").mkdir(parents=True, exist_ok=True)

verbose_level = 2  # 1 : print at each langevin step.
                   # 2 : print at each saddle point iteration.
                 
# Deep Learning            
model_file = "pretrained_models/gyroid_asppnet.pth"
#model_file = "saved_model_49.pth"
#input_data = sio.loadmat("eq_inputs/data_simulation_chin18.0.mat", squeeze_me=True)
input_data = sio.loadmat("temp.mat", squeeze_me=True)

# Simulation Box
nx = [64, 64, 64]
lx = [7.31, 7.31, 7.31]

# Polymer Chain
n_contour = 90
f = 0.4
chi_n = 18.0
chain_model = "Discrete"

# Anderson Mixing
saddle_tolerance_acc = 1e-7
saddle_tolerance = 1e-5
saddle_max_iter = 300
am_n_comp = 1  # W+
am_max_hist= 20
am_start_error = 1e-1
am_mix_min = 0.1
am_mix_init = 0.1

# Langevin Dynamics
langevin_dt = 0.8     # langevin step interval, delta tau*N
langevin_nbar = 10000  # invariant polymerization index
langevin_max_iter = 1

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
    
# random seed for MT19937
np.random.seed(5489)

# Deep Learning model FTS
model = TrainerAndModel()
model.load_state_dict(torch.load(model_file), strict=True)
deepfts = DeepFts(model)
deepfts.eval_mode()

# -------------- print simulation parameters ------------
print("---------- Simulation Parameters ----------");
print("Box Dimension: %d"  % (sb.get_dim()) )
print("Precision: 8")
print("chi_n: %f, f: %f, N: %d" % (pc.get_chi_n(), pc.get_f(), pc.get_n_contour()) )
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

print("wminus and wplus are initialized to random")
w_plus = np.random.normal(0, langevin_sigma, sb.get_n_grid())
w_minus = np.random.normal(0, langevin_sigma, sb.get_n_grid())

w_plus = input_data["w_plus"]# + np.random.normal(0, langevin_sigma, sb.get_n_grid())/2
w_minus = input_data["w_minus"]

# keep the level of field value
sb.zero_mean(w_plus);
sb.zero_mean(w_minus);

# timers
total_saddle_iter = 0
time_dl = 0.0
time_pseudo = 0.0
time_start = time.time()

w_plus_copy = w_plus.copy()

# Find acc with Anderson Mixing
w_plus_acc = w_plus.copy()
find_saddle_point(use_net=False, plot=False, saddle_tolerance=saddle_tolerance_acc)
w_plus_acc = w_plus.copy()

# Run with Anderson Mixing
w_plus = w_plus_copy.copy()
find_saddle_point(use_net=False, plot=True, saddle_tolerance=saddle_tolerance)

# Run with Deep Learning
w_plus = w_plus_copy.copy()
find_saddle_point(use_net=True, plot=True, saddle_tolerance=saddle_tolerance)

