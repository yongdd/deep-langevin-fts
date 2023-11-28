import os
import sys
import time
import numpy as np
from scipy.io import savemat, loadmat
from scipy.ndimage import gaussian_filter
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import deep_langevin_fts

f = 0.4        # A-fraction of major BCP chain, f
eps = 1.0       # a_A/a_B, conformational asymmetry

params = {
    #---------------- Simulation parameters -----------------------------
    "nx":[64, 64, 64],          # Simulation grid numbers
    "lx":[7.31, 7.31, 7.31],    # Simulation box size as a_Ref * N_Ref^(1/2) unit,
                                # where "a_Ref" is reference statistical segment length
                                # and "N_Ref" is the number of segments of reference linear homopolymer chain.

    "chain_model":"discrete",   # "discrete" or "continuous" chain model
    "ds":1/90,                  # Contour step interval, which is equal to 1/N_Ref.

    "segment_lengths":{         # Relative statistical segment length compared to "a_Ref.
        "A":np.sqrt(eps*eps/(eps*eps*f + (1-f))), 
        "B":np.sqrt(    1.0/(eps*eps*f + (1-f))), },

    "chi_n": {"A,B": 18.0},     # Bare interaction parameter, Flory-Huggins params * N_Ref

    "distinct_polymers":[{      # Distinct Polymers
        "volume_fraction":1.0,  # volume fraction of polymer chain
        "blocks":[              # AB diBlock Copolymer
            {"type":"A", "length":f, }, # A-block
            {"type":"B", "length":1-f}, # B-block
        ],},],
        
    "langevin":{                # Langevin Dynamics
        "max_step":500000,      # Langevin steps for simulation
        "dt":8.0,               # Langevin step interval, delta tau*N_Ref
        "nbar":10000,           # Invariant polymerization index, nbar of N_Ref
    },
    
    "recording":{                       # Recording Simulation Data
        "dir":"data_simulation",        # Directory name
        "recording_period":50,        # Period for recording concentrations and fields
        "sf_computing_period":10,       # Period for computing structure function
        "sf_recording_period":200,    # Period for recording structure function
    },

    "saddle":{                # Iteration for the pressure field 
        "max_iter" :100,      # Maximum number of iterations
        "tolerance":1e-4,     # Tolerance of incompressibility 
    },

    "am":{
        "max_hist":20,              # Maximum number of history
        "start_error":5e-1,         # When switch to AM from simple mixing
        "mix_min":0.01,             # Minimum mixing rate of simple mixing
        "mix_init":0.01,            # Initial mixing rate of simple mixing
    },

    "verbose_level":1,      # 1 : Print at each langevin step.
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
        "recording_n_data":3,       # Make 3 training data
        "tolerance":1e-7,           # Tolerance of incompressibility for training data

        # Training GPUs
        "gpus":1,                   # The number of gpus per node
        "num_nodes":1,              # The number of gpu nodes

        # Training Parameters
        "lr":1e-3,                           # Learning rate
        "precision":16,                      # Training precision, [64, 32, 16] = [double, single, mixed] precisions
        "max_epochs":100,                    # The number of epochs
        "model_dir":"saved_model_weights",   # Directory for saved_model_weights

        # Model Parameters
        "features": 64,                      # The number of features for each convolution layer

        # Data Loader
        "batch_size":8,                     # Batch size
        "num_workers":8,                    # The number of workers for data loading
    },
}
# Set random seed
# If you want to obtain different results for each execution, set random_seed=None
random_seed = 12345

# Initialize simulation
simulation = deep_langevin_fts.DeepLangevinFTS(params=params, random_seed=random_seed)

input_data = loadmat("gyroid_equil_chin18.0.mat", squeeze_me=True)
w_A = input_data["w_A"]
w_B = input_data["w_B"]

# Run
simulation.run(initial_fields={"A": w_A, "B": w_B}, max_step=1000, model_file="best_epoch.pth")

# Recording first a few iteration results for debugging and refactoring

# ---------- model file : best_epoch.pth ----------
#        1   -5.255E-16  [ 1.1661655E+01  ]     7.388939934   [9.4629228E-05 ]
# iteration, mass error, total partitions, total energy, incompressibility error (or saddle point error)
# ---------- Run  ----------
# Langevin step:  1
#        4    6.994E-16  [ 7.4120063E+00  ]     6.473788412   [6.4642598E-05 ]
# Langevin step:  2
#        4    2.530E-16  [ 6.0184549E+00  ]     6.723248318   [2.0309188E-05 ]
# Langevin step:  3
#        4   -3.666E-16  [ 5.1485615E+00  ]     6.889040044   [6.9283143E-05 ]
# Langevin step:  4
#        4    1.072E-16  [ 4.5427242E+00  ]     6.999103819   [1.9890984E-05 ]
# Langevin step:  5
#        4   -1.137E-16  [ 4.1020391E+00  ]     7.072875633   [4.2607747E-05 ]
