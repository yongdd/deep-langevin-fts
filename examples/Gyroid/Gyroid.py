import os
import sys
import time
import numpy as np
from scipy.io import savemat, loadmat
from scipy.ndimage.filters import gaussian_filter
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
    "chi_n":18.0,               # Interaction parameter, Flory-Huggins params * N

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

input_data = loadmat("gyroid_equil_chin18.0.mat", squeeze_me=True)

# Run
simulation.run(w_minus=input_data["w_minus"], w_plus=input_data["w_plus"],
    max_step=1000, model_file="gyroid_atr_cas_mish_32.pth")

# Recording first a few iteration results for debugging and refactoring

#        1    5.107E-15  [ 4.5552513E+03  ]     7.388939934   9.4629228E-05
# iteration, mass error, total_partitions, energy_total, error_level
# ---------- Run  ----------
# Langevin step:  1
#        4   -3.331E-15  [ 4.7470327E+03  ]     7.376390976   3.9736048E-05
#        2    5.995E-15  [ 4.6842196E+03  ]     7.356732779   8.1247113E-05
# Langevin step:  2
#        4    6.661E-16  [ 4.8786184E+03  ]     7.359406387   4.0457824E-05
#        2    6.661E-16  [ 4.8160751E+03  ]     7.339170370   8.0894406E-05
# Langevin step:  3
#        4    2.220E-16  [ 5.0064112E+03  ]     7.336935871   2.6789041E-05
#        2    1.044E-14  [ 4.9436785E+03  ]     7.316899864   7.9938932E-05
# Langevin step:  4
#        4    6.217E-15  [ 5.1635702E+03  ]     7.312423003   3.8277041E-05
#        2   -2.776E-15  [ 5.0969040E+03  ]     7.292408623   8.2070880E-05
