import os
import sys
import time
import numpy as np
from scipy.io import savemat, loadmat
from scipy.ndimage.filters import gaussian_filter
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
    "chi_n":17.0,               # Interaction parameter, Flory-Huggins params * N

    "segment_lengths":{         # Relative statistical segment length compared to "a_Ref.
        "A":np.sqrt(eps*eps/(eps*eps*f + (1-f))), 
        "B":np.sqrt(    1.0/(eps*eps*f + (1-f))), },

    "distinct_polymers":[{      # Distinct Polymers
        "volume_fraction":1.0,  # volume fraction of polymer chain
        "blocks":[              # AB diBlock Copolymer
            {"type":"A", "length":f, }, # A-block
            {"type":"B", "length":1-f}, # B-block
        ],},],
        
    "langevin":{                # Langevin Dynamics
        "max_step":500000,      # Langevin steps for simulation
        "dt":0.8,               # Langevin step interval, delta tau*N
        "nbar":10000,           # invariant polymerization index, nbar
    },
    
    "recording":{                       # Recording Simulation Data
        "dir":"data_simulation",        # Directory name
        "recording_period":1000,        # period for recording concentrations and fields
        "sf_computing_period":10,       # period for computing structure function
        "sf_recording_period":10000,    # period for recording structure function
    },

    "saddle":{                # Iteration for the pressure field 
        "max_iter" :100,      # maximum the number of iterations
        "tolerance":1e-4,     # tolerance of incompressibility 
    },

    "am":{
        "max_hist":20,              # Maximum number of history
        "start_error":8e-1,         # When switch to AM from simple mixing
        "mix_min":0.1,              # Minimum mixing rate of simple mixing
        "mix_init":0.1,             # Initial mixing rate of simple mixing
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
        "recording_period":5,       # make training data every 5 langevin steps
        "recording_n_data":3,       # make 3 training data
        "tolerance":1e-7,           # tolerance of incompressibility for training data

        # Training GPUs
        "gpus":1,                   # the number of gpus per node
        "num_nodes":1,              # the number of gpu nodes

        # Training Parameters
        "lr":1e-3,                           # Learning rate
        "precision":16,                      # training precision, [64, 32, 16] = [double, single, mixed] precisions
        "max_epochs":100,                    # the number of epochs
        "model_dir":"saved_model_weights",   # Directory for saved_model_weights

        # Model Parameters
        "features": 32,                      # the number of parameters

        # Data Loader
        "batch_size":8,                     # Batch size
        "num_workers":8,                    # the number of workers for data loading
    },
}
## random seed for MT19937
np.random.seed(5489)

# Initialize calculation
simulation = deep_langevin_fts.DeepLangevinFTS(params=params)

# Run
input_data = loadmat("lamella_equil_chin17.0.mat", squeeze_me=True)
simulation.run(w_minus=input_data["w_minus"], w_plus=input_data["w_plus"],
    max_step=1000, model_file="lamella_atr_cas_mish_32.pth")

# Recording first a few iteration results for debugging and refactoring

# ---------- model file : lamella_atr_cas_mish_32.pth ----------
#        1    4.885E-15  [ 9.0723046E+02  ]     7.686989541   8.5299264E-05
# iteration, mass error, total_partitions, energy_total, error_level
# ---------- Run  ----------
# Langevin step:  1
#        4   -2.776E-15  [ 9.2347260E+02  ]     7.684098073   3.0802194E-05
#        2   -3.553E-15  [ 9.0678642E+02  ]     7.657582210   6.8893412E-05
# Langevin step:  2
#        4   -2.109E-15  [ 9.3777918E+02  ]     7.687300287   1.0764010E-05
#        2   -6.994E-15  [ 9.2026604E+02  ]     7.659518028   7.2876970E-05
# Langevin step:  3
#        4    7.327E-15  [ 9.3812480E+02  ]     7.681600157   1.7492080E-05
#        2    5.107E-15  [ 9.1931015E+02  ]     7.654029517   6.6468549E-05
# Langevin step:  4
#        4    2.887E-15  [ 9.3603484E+02  ]     7.692449298   2.5850354E-05
#        2   -1.332E-15  [ 9.1887446E+02  ]     7.664296313   7.4125324E-05
# Langevin step:  5
#        4    6.661E-16  [ 9.4299783E+02  ]     7.676665165   1.3375244E-05
#        2    4.441E-15  [ 9.2496492E+02  ]     7.649550204   6.7683185E-05