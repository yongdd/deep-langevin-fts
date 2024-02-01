import os
import sys
import time
import numpy as np
from scipy.io import savemat, loadmat
from scipy.ndimage import gaussian_filter
import scipy.ndimage
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import deep_langevin_fts

# Major Simulation params
f = 0.5          # A-fraction of major BCP chain, f
eps = 1.0        # a_A/a_B, conformational asymmetry
n_sc = 50        # the number of side chains
sc_alpha = 0.3   # N_sc/ N_bb
chi_n = 34.0     # Bare interaction parameter, Flory-Huggins params*N_total

def create_bottle_brush(sc_alpha, n_sc, f):

    d = 1/n_sc
    N_BB_A = round(n_sc*f)

    assert(np.isclose(N_BB_A - n_sc*f, 0)), \
        "'n_sc*f' is not an integer."

    blocks = []
    # backbone (A region)
    blocks.append({"type":"A", "length":d/2, "v":0, "u":1})
    for i in range(1,N_BB_A):
        blocks.append({"type":"A", "length":d, "v":i, "u":i+1})

    # backbone (AB junction)
    blocks.append({"type":"A", "length":d/2,   "v":N_BB_A,   "u":N_BB_A+1}) 
    blocks.append({"type":"B", "length":d/2,   "v":N_BB_A+1, "u":N_BB_A+2})

    # backbone (B region)
    for i in range(N_BB_A+1,n_sc):
        blocks.append({"type":"B", "length":d,   "v":i+1, "u":i+2})
    blocks.append({"type":"B", "length":d/2,   "v":n_sc+1, "u":n_sc+2})

    # side chains (A)
    blocks.append({"type":"A", "length":sc_alpha, "v":1, "u":n_sc+3})
    for i in range(1,N_BB_A):
        blocks.append({"type":"A", "length":sc_alpha, "v":i+1, "u":i+n_sc+3})

    # side chains (B)
    for i in range(N_BB_A+1,n_sc):
        blocks.append({"type":"B", "length":sc_alpha, "v":i+1, "u":i+n_sc+2})
    blocks.append({"type":"B", "length":sc_alpha, "v":n_sc+1, "u":2*n_sc+2})
    
    return blocks

blocks = create_bottle_brush(sc_alpha, n_sc, f)
total_alpha = 1 + n_sc*sc_alpha

print("Blocks:", *blocks, sep = "\n")
print(total_alpha)

params = {
    #---------------- Simulation parameters -----------------------------

    # The neural network is trained in 64^3 grids.
    # To use the trained NN in different simulation box size, we change "nx" as well as "lx", fixing grid interval "dx".

    "nx":[64,64,64],        # Simulation grid numbers
    "lx":[6.75,6.75,6.75],  # Simulation box size as a_Ref * N_Ref^(1/2) unit,
                            # where "a_Ref" is reference statistical segment length
                            # and "N_Ref" is the number of segments of reference linear homopolymer chain.
                            
    "aggregate_propagator_computation":True,   # Aggregate multiple propagators when solving diffusion equations for speedup. 
                                               # To obtain concentration of each block, disable this option.

    "chain_model":"discrete",     # "discrete" or "continuous" chain model
    "ds":1/100,                   # Contour step interval, which is equal to 1/N_Ref.
    
    "segment_lengths":{         # Relative statistical segment length compared to "a_Ref.
        "A":np.sqrt(eps*eps/(eps*eps*f + (1-f))), 
        "B":np.sqrt(    1.0/(eps*eps*f + (1-f))), },

    "distinct_polymers":[{      # Distinct Polymers
        "volume_fraction":1.0,  # Volume fraction of polymer chain
        "blocks": blocks,
        },],
    
    "chi_n": {"A,B":chi_n/total_alpha},     # Bare interaction parameter, Flory-Huggins params * N_Ref
    
    "langevin":{                # Langevin Dynamics
        "max_step":200000,      # Langevin steps for simulation
        "dt":0.5,               # Langevin step interval, delta tau*N_Ref
        "nbar":10000,           # Invariant polymerization index, nbar of N_Ref
    },
    
    "recording":{                       # Recording Simulation Data
        "dir":"data_simulation",        # Directory name
        "recording_period":1000,        # Period for recording concentrations and fields
        "sf_computing_period":10,       # Period for computing structure function
        "sf_recording_period":10000,    # Period for recording structure function
    },

    "saddle":{                # Iteration for the pressure field 
        "max_iter" :400,      # Maximum number of iterations
        "tolerance":1e-4,     # Tolerance of incompressibility 
    },

    "am":{
        "max_hist":60,              # Maximum number of history
        "start_error":5e-1,         # When switch to AM from simple mixing
        "mix_min":0.01,             # Minimum mixing rate of simple mixing
        "mix_init":0.01,            # Initial mixing rate of simple mixing
    },

    "verbose_level":1,      # 1 : Print at each Langevin step.
                            # 2 : Print at each saddle point iteration.

    #---------------- Training parameters -----------------------------
    # 1. If you plan to use multiple GPUs for training, edit "gpus".
    #   To obtain the same training results using multiple GPUs, 
    #   you need to change `batch_size` so that `gpus` * `batch_size` does not change.
    #   For example, if you use 4 GPUs, set `gpus=4` and `batch_size=8`, 
    #   which is effectively the same as setting `gpus=1` and `batch_size=32`.

    "training":{           
        # Training Data         
        "data_dir":"data_training", # Directory name
        "max_step":10000,           # Langevin steps for collecting training data
        "recording_period":5,       # Make training data every 5 Langevin steps
        "recording_n_data":4,       # Make 3 training data
        "tolerance":1e-7,           # Tolerance of incompressibility for training data

        # Training GPUs
        "gpus":1,                   # The number of gpus per node
        "num_nodes":1,              # The number of gpu nodes

        # Training Parameters
        "lr":1e-3,                           # Learning rate
        "precision":"16-mixed",              # Training precision, [64, 32, 16_mixed]
        "max_epochs":200,                    # The number of epochs
        "model_dir":"saved_model_weights",   # Directory for saved_model_weights

        # Model Parameters
        "features":32,                       # The number of features for each convolution layer

        # Data Loader
        "batch_size":8,                      # Batch size
        "num_workers":8,                     # The number of workers for data loading
    },
}
# Set random seed
# If you want to obtain different results for each execution, set random_seed=None
random_seed = 12345

# Set initial fields
# print("w_A and w_B are initialized to gyroid phase")
# input_data = loadmat("LastTrainingLangevinStep.mat", squeeze_me=True)
# w_A = input_data["w_A"]
# w_B = input_data["w_B"]

# # Interpolate input data on params["nx"], if necessary
# w_A = scipy.ndimage.zoom(np.reshape(w_A, input_data["nx"]), params["nx"]/input_data["nx"])
# w_B = scipy.ndimage.zoom(np.reshape(w_B, input_data["nx"]), params["nx"]/input_data["nx"])
# initial_fields={"A": w_A, "B": w_B}

# Initialize calculation
simulation = deep_langevin_fts.DeepLangevinFTS(params=params, random_seed=random_seed)

# # Make training data
# # After training data are generated, the field configurations of the last Langevin step will be saved with the file name "LastTrainingLangevinStep.mat".
# simulation.make_training_dataset(initial_fields=initial_fields, final_fields_configuration_file_name="LastTrainingLangevinStep.mat")

# # Train model
# simulation.train_model()

# # Find best epoch
# # The best neural network weights will be saved with the file name "best_epoch.pth".
input_data = loadmat("LastTrainingLangevinStep.mat", squeeze_me=True)
w_A = input_data["w_A"]
w_B = input_data["w_B"]

# # # Interpolate input data on params["nx"], if necessary
# # w_A = scipy.ndimage.zoom(np.reshape(w_A, input_data["nx"]), params["nx"]/input_data["nx"])
# # w_B = scipy.ndimage.zoom(np.reshape(w_B, input_data["nx"]), params["nx"]/input_data["nx"])
initial_fields={"A": w_A, "B": w_B}

# simulation.find_best_epoch(initial_fields=initial_fields, best_epoch_file_name="best_epoch.pth")

# Run
simulation.run(initial_fields=initial_fields, max_step=1000, model_file="best_epoch.pth")

# Recording first a few iteration results for debugging and refactoring

# ---------- model file : best_epoch.pth ----------
#        1   -8.313E-16  [ 1.6366941E+02  ]     4.727348048   [9.9744451E-08 ]
# iteration, mass error, total partitions, total energy, incompressibility error (or saddle point error)
# ---------- Run  ----------
# Langevin step:  1
#        8    2.934E-16  [ 1.3613622E+02  ]     4.049023313   [7.5321997E-05 ]
# Langevin step:  2
#        6   -5.259E-16  [ 1.3071677E+02  ]     4.196220288   [6.0146113E-05 ]
# Langevin step:  3
#        6   -2.052E-15  [ 1.3048926E+02  ]     4.313089645   [4.3308454E-05 ]
# Langevin step:  4
#        7    6.190E-17  [ 1.3714328E+02  ]     4.405333796   [9.5100680E-05 ]
# Langevin step:  5
#        6   -2.951E-15  [ 1.4159085E+02  ]     4.473910838   [4.3779476E-05 ]