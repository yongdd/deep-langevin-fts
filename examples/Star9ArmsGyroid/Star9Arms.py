import os
import sys
import time
import numpy as np
from scipy.io import savemat, loadmat
from scipy.ndimage import gaussian_filter
import scipy.ndimage
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import deep_langevin_fts

f = 0.4        # A-fraction of major BCP chain, f
eps = 1.0       # a_A/a_B, conformational asymmetry

params = {
    #---------------- Simulation parameters -----------------------------

    # The neural network is trained in 64^3 grids.
    # To use the trained NN in different simulation box size, we change "nx" as well as "lx", fixing grid interval "dx".

    "nx":[64,64,64],        # Simulation grid numbers
    "lx":[7.37, 7.37, 7.37],
                            # Simulation box size as a_Ref * N_Ref^(1/2) unit,
                            # where "a_Ref" is reference statistical segment length
                            # and "N_Ref" is the number of segments of reference linear homopolymer chain.

    "chain_model":"discrete",      # "discrete" or "continuous" chain model
    "ds":1/100,                    # Contour step interval, which is equal to 1/N_Ref.
    
    "segment_lengths":{            # Relative statistical segment length compared to "a_Ref.
        "A":np.sqrt(eps*eps/(eps*eps*f + (1-f))), 
        "B":np.sqrt(    1.0/(eps*eps*f + (1-f))), },

    "chi_n": {"A,B": 12.5},     # Bare interaction parameter, Flory-Huggins params * N_Ref

    "distinct_polymers":[{      # Distinct Polymers
        "volume_fraction":1.0,  # Volume fraction of polymer chain
        "blocks":[              # 9-arm star-shaped AB Copolymer
            {"type":"A", "length":f,   "v":0, "u":1},    # A-block
            {"type":"A", "length":f,   "v":0, "u":2},    # A-block
            {"type":"A", "length":f,   "v":0, "u":3},    # A-block
            {"type":"A", "length":f,   "v":0, "u":4},    # A-block
            {"type":"A", "length":f,   "v":0, "u":5},    # A-block
            {"type":"A", "length":f,   "v":0, "u":6},    # A-block
            {"type":"A", "length":f,   "v":0, "u":7},    # A-block
            {"type":"A", "length":f,   "v":0, "u":8},    # A-block
            {"type":"A", "length":f,   "v":0, "u":9},    # A-block
            {"type":"B", "length":1-f, "v":1, "u":10},    # B-block
            {"type":"B", "length":1-f, "v":2, "u":11},    # B-block
            {"type":"B", "length":1-f, "v":3, "u":12},    # B-block
            {"type":"B", "length":1-f, "v":4, "u":13},    # B-block
            {"type":"B", "length":1-f, "v":5, "u":14},    # B-block
            {"type":"B", "length":1-f, "v":6, "u":15},    # B-block
            {"type":"B", "length":1-f, "v":7, "u":16},    # B-block
            {"type":"B", "length":1-f, "v":8, "u":17},    # B-block
            {"type":"B", "length":1-f, "v":9, "u":18},    # B-block
        ],},],
    
    "langevin":{                # Langevin Dynamics
        "max_step":200000,      # Langevin steps for simulation
        "dt":8.0,               # Langevin step interval, delta tau*N_Ref
        "nbar":10000,           # Invariant polymerization index, nbar of N_Ref
    },
    
    "recording":{                       # Recording Simulation Data
        "dir":"data_simulation",        # Directory name
        "recording_period":1000,        # Period for recording concentrations and fields
        "sf_computing_period":10,       # Period for computing structure function
        "sf_recording_period":10000,    # Period for recording structure function
    },

    "saddle":{                # Iteration for the pressure field 
        "max_iter" :200,      # Maximum the number of iterations
        "tolerance":1e-4,     # Tolerance of incompressibility 
    },

    "compressor":{
        "name":"am",                # Anderson Mixing
        # "name":"lr",                # Linear Response
        # "name":"lram",              # Linear Response + Anderson Mixing
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
        "features":64,                       # The number of features for each convolution layer

        # Data Loader
        "batch_size":8,                      # Batch size
        "num_workers":8,                     # The number of workers for data loading
    },
}
# Set random seed
# If you want to obtain different results for each execution, set random_seed=None
random_seed = 12345

# # Set initial fields
# print("w_A and w_B are initialized to gyroid phase")
# input_data = loadmat("fields_500000.mat", squeeze_me=True)
# w_A = input_data["w_A"]
# w_B = input_data["w_B"]

# # # Interpolate input data on params["nx"], if necessary
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

# Find best epoch
# The best neural network weights will be saved with the file name "best_epoch.pth".
input_fields_data = loadmat("LastTrainingLangevinStep.mat", squeeze_me=True)
w_A = input_fields_data["w_A"]
w_B = input_fields_data["w_B"]
initial_fields={"A": w_A, "B": w_B}
# simulation.find_best_epoch(initial_fields=initial_fields, best_epoch_file_name="best_epoch.pth")

# Run
simulation.run(initial_fields=initial_fields, max_step=1000, model_file="best_epoch.pth")

# Recording first a few iteration results for debugging and refactoring

# ---------- model file : best_epoch.pth ----------
#        1   -4.751E-16  [ 2.8531735E+03  ]     5.838952661   [9.3053987E-08 ]
# iteration, mass error, total partitions, Hamiltonian, incompressibility error (or saddle point error)
# ---------- Run  ----------
# Langevin step:  1
#        8    1.092E-15  [ 6.6053031E+07  ]     4.584050430   [9.8126311E-05 ]
# Langevin step:  2
#        9   -2.901E-16  [ 3.1274564E+08  ]     5.769369798   [4.4055584E-05 ]
# Langevin step:  3
#        8   -9.450E-17  [ 3.1800963E+08  ]     5.808993716   [9.3180371E-05 ]
# Langevin step:  4
#        4 This neural network could not reduce the incompressibility error (or saddle point error), and it was switched to Anderson mixing.
#       69    4.809E-16  [ 2.0384389E+09  ]     5.827503021   [9.4180187E-05 ]
# Langevin step:  5
#        8   -3.471E-16  [ 2.4621924E+08  ]     5.822811062   [7.7000321E-05 ]