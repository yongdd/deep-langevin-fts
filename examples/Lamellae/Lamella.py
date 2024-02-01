import os
import sys
import time
import numpy as np
from scipy.io import savemat, loadmat
from scipy.ndimage import gaussian_filter
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import deep_langevin_fts

# # Major Simulation params
f = 1/2         # A-fraction of major BCP chain, f
eps = 1.0       # a_A/a_B, conformational asymmetry

params = {
    #---------------- Simulation parameters -----------------------------
    "nx":[40, 40, 40],          # Simulation grid numbers
    "lx":[4.36, 4.36, 4.36],    # Simulation box size as a_Ref * N_Ref^(1/2) unit,
                                # where "a_Ref" is reference statistical segment length
                                # and "N_Ref" is the number of segments of reference linear homopolymer chain.

    "chain_model":"discrete",   # "discrete" or "continuous" chain model
    "ds":1/90,                  # Contour step interval, which is equal to 1/N_Ref.

    "segment_lengths":{         # Relative statistical segment length compared to "a_Ref.
        "A":np.sqrt(eps*eps/(eps*eps*f + (1-f))), 
        "B":np.sqrt(    1.0/(eps*eps*f + (1-f))), },

    "chi_n": {"A,B":17.0},     # Bare interaction parameter, Flory-Huggins params * N_Ref

    "distinct_polymers":[{      # Distinct Polymers
        "volume_fraction":1.0,  # Volume fraction of polymer chain
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
        "recording_period":1000,        # Period for recording concentrations and fields
        "sf_computing_period":10,       # Period for computing structure function
        "sf_recording_period":10000,    # Period for recording structure function
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
        "recording_n_data":4,       # Make 3 training data
        "tolerance":1e-7,           # Tolerance of incompressibility for training data

        # Training GPUs
        "gpus":1,                   # The number of gpus per node
        "num_nodes":1,              # The number of gpu nodes

        # Training Parameters
        "lr":1e-3,                           # Learning rate
        "precision":"16-mixed",              # Training precision, [64, 32, 16_mixed]
        "max_epochs":100,                    # The number of epochs
        "model_dir":"saved_model_weights",   # Directory for saved_model_weights

        # Model Parameters
        "features":32,                      # The number of features for each convolution layer

        # Data Loader
        "batch_size":8,                     # Batch size
        "num_workers":8,                    # The number of workers for data loading
    },
}
# Set random seed
# If you want to obtain different results for each execution, set random_seed=None
random_seed = 12345

# # Set initial fields
# input_data = loadmat("LamellaInput.mat", squeeze_me=True)
# w_A = input_data["w_A"]
# w_B = input_data["w_B"]
# initial_fields={"A": w_A, "B": w_B}

# Initialize calculation
simulation = deep_langevin_fts.DeepLangevinFTS(params=params, random_seed=random_seed)

# # Generate training data
# # After training data are generated, the field configurations of the last Langevin step will be saved with the file name "LastTrainingLangevinStep.mat".
# simulation.make_training_dataset(initial_fields=initial_fields, final_fields_configuration_file_name="LastTrainingLangevinStep.mat")

# # Train model
# simulation.train_model()

# # Find best epoch
# # The best neural network weights will be saved with the file name "best_epoch.pth".
# input_fields_data = loadmat("LastTrainingLangevinStep.mat", squeeze_me=True)
# w_A = input_fields_data["w_A"]
# w_B = input_fields_data["w_B"]
# initial_fields={"A": w_A, "B": w_B}
# simulation.find_best_epoch(initial_fields=initial_fields, best_epoch_file_name="best_epoch.pth")

# Run
input_data = loadmat("lamella_equil_chin17.0.mat", squeeze_me=True)
w_A = input_data["w_A"]
w_B = input_data["w_B"]
initial_fields={"A": w_A, "B": w_B}
simulation.run(initial_fields=initial_fields, max_step=1000, model_file="best_epoch.pth")

# Recording first a few iteration results for debugging and refactoring

# ---------- model file : best_epoch.pth ----------
#        1    4.784E-16  [ 1.0946068E+01  ]     7.686989541   [8.5299264E-05 ]
# iteration, mass error, total partitions, total energy, incompressibility error (or saddle point error)
# ---------- Run  ----------
# Langevin step:  1
#        5   -1.530E-16  [ 1.0430770E+01  ]     6.672228868   [8.9467214E-06 ]
# Langevin step:  2
#        4   -5.588E-16  [ 1.1028248E+01  ]     7.051253962   [4.6317405E-05 ]
# Langevin step:  3
#        4    1.513E-16  [ 1.1861312E+01  ]     7.264916294   [4.7246821E-05 ]
# Langevin step:  4
#        4   -9.282E-16  [ 1.2623571E+01  ]     7.386787832   [4.9695800E-05 ]
# Langevin step:  5
#        4   -7.808E-16  [ 1.2347065E+01  ]     7.476911481   [3.1146655E-05 ]
