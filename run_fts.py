import sys
import os
import time
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import *
from langevinfts import *
from trainer_and_model import *
from deep_fts import *

def save_data(path, name, langevin_step, idx, w_minus, g_plus, w_plus_diff):
    out_file_name = "%s_%06d_%03d.npz" % (name, langevin_step, idx)
    np.savez( os.path.join(path, out_file_name),
        nx=nx, lx=lx, N=n_contour, f=pc.get_f(), chi_n=pc.get_chi_n(),
        polymer_model=chain_model, n_bar=langevin_nbar,
        w_minus=w_minus.astype(np.float16),
        g_plus=g_plus.astype(np.float16),
        w_plus_diff=w_plus_diff.astype(np.float16))

def find_saddle_point(tolerance, net=None):
    # assign large initial value for the energy and error
    energy_total = 1e20
    error_level = 1e20

    # reset Anderson mixing module
    am.reset_count()

    global phi_a
    global phi_b
    global w_plus
    global w_minus
 
    time_neural_net = 0.0
    time_pseudo = 0.0
    
    is_net_diverged = False
    # saddle point iteration begins here
    for saddle_iter in range(1,saddle_max_iter+1):
       
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
        if(is_net_diverged == False and error_level >= old_error_level):
           is_net_diverged = True

        # print iteration # and error levels
        if(verbose_level == 2 or
         verbose_level == 1 and
         (error_level < tolerance or saddle_iter == saddle_max_iter)):
             
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
        if(error_level < tolerance):
            break;
       
        if (net and not is_net_diverged):
            # calculte new fields using neural network
            time_d_start = time.time()
            w_plus_diff = net.generate_w_plus(w_minus, g_plus, sb.get_nx()[:sb.get_dim()])
            w_plus += w_plus_diff
            sb.zero_mean(w_plus)
            time_neural_net += time.time() - time_d_start
        else:
            # calculte new fields using simple and Anderson mixing
            w_plus_out = w_plus + g_plus 
            sb.zero_mean(w_plus_out)
            am.caculate_new_fields(w_plus, w_plus_out, g_plus, old_error_level, error_level);
    return time_pseudo, time_neural_net, saddle_iter, error_level, is_net_diverged

def collect_training_data(langevin_max_iter, net=None):
    
    global phi_a
    global phi_b
    global w_plus
    global w_minus
    
    for langevin_step in range(1, langevin_max_iter+1):
        print("Langevin step: ", langevin_step)
        # update w_minus
        normal_noise = np.random.normal(0.0, langevin_sigma, sb.get_n_grid())
        g_minus = phi_a-phi_b + 2*w_minus/pc.get_chi_n()
        w_minus += -g_minus*langevin_dt + normal_noise
        sb.zero_mean(w_minus)

        # find saddle point
        find_saddle_point(tolerance=saddle_tolerance, net=None)
        w_plus_tol = w_plus.copy()
        g_plus_tol = phi_a + phi_b - 1.0

        # find more accurate saddle point
        find_saddle_point(tolerance=saddle_tolerance_ref, net=None)
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
                save_data(data_path_training, "fields_1st_%d" % np.round(chi_n*100), langevin_step, idx, 
                w_minus, g_plus, w_plus_ref-w_plus_noise)
            
def train(model, train_dir, max_epochs):

    # training data    
    train_dataset = FtsDataset(train_dir)
    train_loader = DataLoader(train_dataset, batch_size=8, num_workers=4)
    print("len(train_dataset)", len(train_dataset))
    
    # training
    trainer = pl.Trainer(
            gpus=1, num_nodes=1, max_epochs=max_epochs, precision=16,
            strategy=DDPPlugin(find_unused_parameters=False),
            benchmark=True, log_every_n_steps=5)
            
    trainer.fit(model, train_loader, None)

# -------------- major parameters ------------

# Deep Learning
train_new_model = False #True 
use_pretrained_model = True #False
pretrained_model_file = "pretrained_models/gyroid_atrnet.pth"
#pretrained_model_file = "saved_model_weights/epoch_199.pth"

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
langevin_dt = 3.2       # langevin step interval, delta tau*N
langevin_nbar = 10000   # invariant polymerization index
langevin_recording_period = 1000
langevin_max_iter = 200000

# Structure Factor
sf_computing_period = 10
sf_recording_period = 10000

#------------- non-polymeric parameters -----------------------------

# OpenMP environment variables 
os.environ["MKL_NUM_THREADS"] = "1"  # always 1
os.environ["OMP_STACKSIZE"] = "1G"
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"  # 0, 1 or 2
torch.set_num_threads(1)

# Cuda environment variables 
os.environ["CUDA_VISIBLE_DEVICES"]= "1"

# Distributed Data Parallel environment variables 
os.environ["PL_TORCH_DISTRIBUTED_BACKEND"]="gloo" #nccl or gloo

data_path_training = "data_training"
data_path_simulation = "data_simulation_dt_32"
pathlib.Path(data_path_training  ).mkdir(parents=True, exist_ok=True)
pathlib.Path(data_path_simulation).mkdir(parents=True, exist_ok=True)

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

# create deep learning model
model = TrainerAndModel()
if (use_pretrained_model):
    model.load_state_dict(torch.load(pretrained_model_file), strict=True)
if( train_new_model or use_pretrained_model ):
    net = DeepFts(model)
    net.eval_mode()
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

print("wminus and wplus are initialized to random")
w_plus = np.random.normal(0, langevin_sigma, sb.get_n_grid())
w_minus = np.random.normal(0, langevin_sigma, sb.get_n_grid())

input_data = np.load("GyroidScftInput.npz")
w_plus = (input_data["w"][0] + input_data["w"][1])/2
w_minus = (input_data["w"][0] - input_data["w"][1])/2

# keep the level of field value
sb.zero_mean(w_plus);
sb.zero_mean(w_minus);

# find saddle point 
find_saddle_point(tolerance=saddle_tolerance)

#------------------ run ----------------------
print("iteration, mass error, total_partition, energy_total, error_level")
if( train_new_model ):
    print("---------- Data Collection ----------")
    collect_training_data(langevin_max_iter=langevin_max_iter_training)
                
    print("---------- Training ----------")
    net.train_mode()
    train(model, data_path_training, 100)
    net.eval_mode()

#input_data = np.load("GyroidFts.npz")
#w_plus = input_data["w_plus"]
#w_minus = input_data["w_minus"]

print("---------- Run  ----------")
sf_average = np.zeros_like(np.fft.rfftn(np.reshape(w_minus, sb.get_nx()[:sb.get_dim()])),np.float64)

# init timer
total_saddle_iter = 0
total_time_neural_net = 0.0
total_time_pseudo = 0.0
total_net_diverged = 0
time_start = time.time()
for langevin_step in range(1, langevin_max_iter+1):
    
    print("Langevin step: ", langevin_step)
    # update w_minus: predict step
    w_minus_copy = w_minus.copy()
    normal_noise = np.random.normal(0.0, langevin_sigma, sb.get_n_grid())
    lambda1 = phi_a-phi_b + 2*w_minus/pc.get_chi_n()
    w_minus += -lambda1*langevin_dt + normal_noise
    sb.zero_mean(w_minus)
    (time_pseudo, time_neural_net, saddle_iter, error_level, is_net_diverged) \
        = find_saddle_point(tolerance = saddle_tolerance, net=net)
    total_time_pseudo += time_pseudo
    total_time_neural_net += time_neural_net
    total_saddle_iter += saddle_iter
    if (is_net_diverged): total_net_diverged += 1
    if (np.isnan(error_level) or error_level >= saddle_tolerance):
        print("Could not satisfy tolerance")
        break;

    # update w_minus: correct step 
    lambda2 = phi_a-phi_b + 2*w_minus/pc.get_chi_n()
    w_minus = w_minus_copy - 0.5*(lambda1+lambda2)*langevin_dt + normal_noise
    sb.zero_mean(w_minus)
    (time_pseudo, time_neural_net, saddle_iter, error_level, is_net_diverged) \
         = find_saddle_point(tolerance = saddle_tolerance, net=net)
    total_time_pseudo += time_pseudo
    total_time_neural_net += time_neural_net
    total_saddle_iter += saddle_iter
    if (is_net_diverged): total_net_diverged += 1
    if (np.isnan(error_level) or error_level >= saddle_tolerance):
        print("Could not satisfy tolerance")
        break;

    # calcaluate structure factor
    if ( langevin_step % sf_computing_period == 0):
        sf_average += np.absolute(np.fft.rfftn(np.reshape(w_minus, sb.get_nx()[:sb.get_dim()]))/sb.get_n_grid())**2

    # save structure factor
    if ( langevin_step % sf_recording_period == 0):
        sf_average *= sf_computing_period/sf_recording_period* \
              sb.get_volume()*np.sqrt(langevin_nbar)/pc.get_chi_n()**2
        sf_average -= 1.0/(2*pc.get_chi_n())
        mdic = {"dim":sb.get_dim(), "nx":sb.get_nx(), "lx":sb.get_lx(),
        "N":pc.get_n_contour(), "f":pc.get_f(), "chi_n":pc.get_chi_n(),
        "chain_model":chain_model,
        "langevin_dt":langevin_dt, "nbar":langevin_nbar,
        "structure_factor":sf_average}
        savemat(os.path.join(data_path_simulation, "structure_factor_%06d.mat" % (langevin_step)), mdic)
        sf_average[:,:,:] = 0.0

    # save simulation data
    if( (langevin_step) % langevin_recording_period == 0 ):
        mdic = {"dim":sb.get_dim(), "nx":sb.get_nx(), "lx":sb.get_lx(),
        "N":pc.get_n_contour(), "f":pc.get_f(), "chi_n":pc.get_chi_n(),
        "chain_model":chain_model,
        "langevin_dt":langevin_dt, "nbar":langevin_nbar,
        "w_plus":w_plus, "w_minus":w_minus, "phi_a":phi_a, "phi_b":phi_b}
        savemat(os.path.join(data_path_simulation, "fields_%06d.mat" % (langevin_step)), mdic)

# estimate execution time
print( "Total iterations for saddle points: %d, iter per step: %f" %
    (total_saddle_iter, total_saddle_iter/langevin_max_iter) )
time_duration = time.time() - time_start; 
print( "Total time: %f, time per step: %f" %
    (time_duration, time_duration/langevin_max_iter) )
print( "Pseudo time ratio: %f, deep learning time ratio: %f" %
    (total_time_pseudo/time_duration, total_time_neural_net/time_duration) )
print( "Total deep learning diverged: %d" % (total_net_diverged) )
