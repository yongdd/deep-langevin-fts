import os
import sys
import time
import numpy as np
from scipy.io import savemat, loadmat
from scipy.ndimage.filters import gaussian_filter
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import deep_langevin_fts

# # Major Simulation params
f = 1/3         # A-fraction of major BCP chain, f
eps = 1.0       # a_A/a_B, conformational asymmetry

params = {
    #---------------- Simulation parameters -----------------------------
    "nx":[64, 48, 48],                      # Simulation grid numbers
    "lx":[6.4, 5.52, 5.52*np.sqrt(3/4)],    # Simulation box size as a_Ref * N_Ref^(1/2) unit,
                                            # where "a_Ref" is reference statistical segment length
                                            # and "N_Ref" is the number of segments of reference linear homopolymer chain.

    "chain_model":"discrete",   # "discrete" or "continuous" chain model
    "ds":1/90,                  # Contour step interval, which is equal to 1/N_Ref.
    "chi_n":20.5,               # Interaction parameter, Flory-Huggins params * N

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
input_data = loadmat("cylinder_equil_chin20.5.mat", squeeze_me=True)
simulation.run(w_minus=input_data["w_minus"], w_plus=input_data["w_plus"],
    max_step=1000, model_file="cylinder_atr_cas_mish_32.pth")

# Recording first a few iteration results for debugging and refactoring

# ---------- model file : cylinder_atr_cas_mish_32.pth ----------
#        1    6.661E-16  [ 7.3750239E+03  ]     8.417113532   8.0769481E-05
# iteration, mass error, total_partitions, energy_total, error_level
# ---------- Run  ----------
# Langevin step:  1
#        4    9.992E-15  [ 7.5308723E+03  ]     8.442976418   5.8338562E-05
#        2   -1.554E-15  [ 7.4489895E+03  ]     8.421447119   8.0796068E-05
# Langevin step:  2
#        4   -7.216E-15  [ 7.3991213E+03  ]     8.449895329   4.5905170E-05
#        2    4.441E-15  [ 7.3317660E+03  ]     8.428296185   8.0572691E-05
# Langevin step:  3
#        4    5.107E-15  [ 7.4409799E+03  ]     8.455470299   4.6926940E-05
#        2    6.883E-15  [ 7.3664257E+03  ]     8.433817589   8.1657240E-05
# Langevin step:  4
#        4    1.776E-15  [ 7.5046539E+03  ]     8.462587250   4.7565468E-05
#        2   -3.331E-15  [ 7.4315004E+03  ]     8.441023753   8.4371393E-05
# Langevin step:  5
#        4    9.548E-15  [ 7.4244808E+03  ]     8.447742795   3.8510529E-05
#        2    6.883E-15  [ 7.3627226E+03  ]     8.427063377   8.2354338E-05