import os
import sys
import time
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from langevinfts import *
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from saddle_net import *

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
 
    # saddle point iteration begins here
    for saddle_iter in range(1,saddle_max_iter+1):

        if (plot==True and
            (use_net==False and saddle_iter==21 or
             use_net==True  and (saddle_iter==1 or saddle_iter==2))):
            record_mat = True
        else:
            record_mat = False

        # for the given fields find the polymer statistics
        time_p_start = time.time()
        phi_a, phi_b, Q = pseudo.find_phi(q1_init, q2_init,
            w_plus+w_minus, w_plus-w_minus)
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
         (error_level < saddle_tolerance or saddle_iter == saddle_max_iter)):
             
            # calculate the total energy
            energy_old = energy_total
            energy_total  = -np.log(Q/sb.get_volume())
            energy_total += sb.inner_product(w_minus,w_minus)/pc.get_chi_n()/sb.get_volume()
            energy_total -= sb.integral(w_plus)/sb.get_volume()

            # check the mass conservation
            mass_error = sb.integral(phi_plus)/sb.get_volume() - 1.0
            print("%8d %12.3E %15.7E %13.9f %13.9f" %
                (saddle_iter, mass_error, Q, energy_total, error_level))
        # conditions to end the iteration
        if error_level < saddle_tolerance :
            total_saddle_iter += saddle_iter
            break
        
        wpd = w_plus_ref - w_plus
        sb.zero_mean(wpd)

        if use_net:
            # predict new field using neural network
            time_d_start = time.time()
            w_plus_diff = model.predict_w_plus(w_minus, g_plus, sb.get_nx()[-sb.get_dim():])
            w_plus += w_plus_diff
            sb.zero_mean(w_plus)
            time_dl += time.time() - time_d_start
            
            wpd_gen = w_plus_diff.copy()
            sb.zero_mean(wpd_gen)
            
            target = torch.tensor(wpd)
            output = torch.tensor(wpd_gen)
            
            #print("mean1", torch.mean((target - output)**2))
        else:
            # calculte new fields using simple and Anderson mixing
            w_plus_out = w_plus + g_plus 
            sb.zero_mean(w_plus_out)
            w_plus_b_am = w_plus.copy()
            am.caculate_new_fields(w_plus, w_plus_out, g_plus, old_error_level, error_level)
            wpd_gen = w_plus.copy() - w_plus_b_am

        if record_mat:
            X = np.linspace(0, lx[0], nx[0], endpoint=False)
            wm = w_minus
            wp = w_plus
            gp = g_plus
            
            sb.zero_mean(wpd)
            sb.zero_mean(wpd_gen)
                               
            #print(np.mean(wpd), np.mean(wpd_gen))
            fig, axes = plt.subplots(2,2, figsize=(20,15))

            plot_x1 = 0
            plot_x2 = nx[0]
            axes[0,0].plot(X, wm     [plot_x1:plot_x2], )
            axes[0,1].plot(X, wp     [plot_x1:plot_x2], )
            axes[1,0].plot(X, gp     [plot_x1:plot_x2], )
            axes[1,1].plot(X, wpd    [plot_x1:plot_x2], )
            axes[1,1].plot(X, wpd_gen[plot_x1:plot_x2], )
            #axes[1,0].plot(X, wpd[:nx[0]]/np.std(gp)/20)
            #axes[1,0].plot(X, wpd_gen[:nx[0]]/np.std(gp)/20)

            #plt.ylim([-0.15, 0.15])
            plt.subplots_adjust(left=0.2,bottom=0.2,
                                top=0.8,right=0.8,
                                wspace=0.2, hspace=0.2)
            if (use_net):
                plt.savefig('difference_%s_%02d.png' % (use_net, saddle_iter))
            else:
                plt.savefig('difference_%s_%02d.png' % (use_net, saddle_iter))
            plt.close()

            mdic = {"dim":sb.get_dim(), "nx":sb.get_nx(), "lx":sb.get_lx(),
                "N":pc.get_n_segment(), "f":pc.get_f(), "chi_n":pc.get_chi_n(),
                "chain_model":chain_model,
                "langevin_dt":langevin_dt, "nbar":langevin_nbar,
                "w_plus":w_plus.copy(), "w_minus":w_minus.copy(),
                "wpd":wpd.copy(), "wpd_gen":wpd_gen.copy(),
                "phi_a":phi_a.copy(), "phi_b":phi_b.copy()}
            sio.savemat("difference_%s_%02d.mat" % (use_net, saddle_iter), mdic)

            if(use_net==False):
                return mdic

# -------------- simulation parameters ------------

#os.environ["CUDA_VISIBLE_DEVICES"]= "0"
verbose_level = 2  # 1 : print at each langevin step.
                   # 2 : print at each saddle point iteration.
                 
# Deep Learning            
model_file = "../pretrained_models/gyroid_atr_par_32.pth"
input_data = sio.loadmat("../eq_inputs/data_simulation_chin18.0.mat", squeeze_me=True)

# Simulation Box
nx = input_data['nx'].tolist()
lx = input_data['lx'].tolist()

# Polymer Chain
n_segment = input_data['N']
f = input_data['f']
chi_n = input_data['chi_n']
chain_model = input_data['chain_model']
epsilon = 1.0   # a_A/a_B, conformational asymmetry

# Read initial fields
w_plus = input_data["w_plus"]
w_minus = input_data["w_minus"]

# Anderson Mixing
saddle_tolerance     = 1e-4
saddle_tolerance_ref = 1e-7
saddle_max_iter = 100
am_n_var = np.prod(nx) # W+
am_max_hist= 20
am_start_error = 1e-1
am_mix_min = 0.1
am_mix_init = 0.1

# Langevin Dynamics
langevin_dt = input_data['langevin_dt']  # langevin step interval, delta tau*N
langevin_nbar = input_data['nbar']       # invariant polymerization index

# -------------- initialize ------------
# choose platform among [cuda, cpu-mkl, cpu-fftw]
factory = PlatformSelector.create_factory("cuda")

# create instances
pc     = factory.create_polymer_chain(f, n_segment, chi_n, chain_model, epsilon)
sb     = factory.create_simulation_box(nx, lx)
pseudo = factory.create_pseudo(sb, pc)
am     = factory.create_anderson_mixing(am_n_var,
            am_max_hist, am_start_error, am_mix_min, am_mix_init)

# standard deviation of normal noise for single segment
langevin_sigma = np.sqrt(2*langevin_dt*sb.get_n_grid()/ 
    (sb.get_volume()*np.sqrt(langevin_nbar)))
    
# random seed for MT19937
np.random.seed(5489)

# deep learning
model = SaddleNet(dim=3, features=32)
model.load_state_dict(torch.load(model_file), strict=True)

# -------------- print simulation parameters ------------
print("---------- Simulation Parameters ----------")
print("Box Dimension: %d"  % (sb.get_dim()) )
print("chi_n: %f, f: %f, N: %d" % (pc.get_chi_n(), pc.get_f(), pc.get_n_segment()) )
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
q1_init = np.ones(sb.get_n_grid(), dtype=np.float64)
q2_init = np.ones(sb.get_n_grid(), dtype=np.float64)

normal_noise = np.random.normal(0.0, langevin_sigma, sb.get_n_grid())
phi_a, phi_b, QQ = pseudo.find_phi(q1_init, q2_init,
    w_plus+w_minus, w_plus-w_minus)
lambda1 = phi_a-phi_b + 2*w_minus/pc.get_chi_n()
w_minus += -lambda1*langevin_dt + normal_noise

# keep the level of field value
sb.zero_mean(w_plus)
sb.zero_mean(w_minus)

# timers
total_saddle_iter = 0
time_dl = 0.0
time_pseudo = 0.0
time_start = time.time()

w_plus_copy = w_plus.copy()

# Find an Accurate Saddle Point with Anderson Mixing
w_plus_ref = w_plus.copy()
find_saddle_point(use_net=False, plot=False, saddle_tolerance=saddle_tolerance_ref)
w_plus_ref = w_plus.copy()

# Run with Anderson Mixing
w_plus = w_plus_copy.copy()
mdic = find_saddle_point(use_net=False, plot=True, saddle_tolerance=saddle_tolerance)

# Run with Deep Learning
w_plus = mdic['w_plus'] - mdic['wpd_gen'] 
w_minus= mdic['w_minus']
find_saddle_point(use_net=True, plot=True, saddle_tolerance=1e-5)
