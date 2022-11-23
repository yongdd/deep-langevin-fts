import os
import time
import numpy as np
from scipy.io import savemat, loadmat
from scipy.ndimage.filters import gaussian_filter
import deep_langevin_fts

f = 0.4         # A-fraction of major BCP chain, f
eps = 1.5       # a_A/a_B, conformational asymmetry

params = {
    #---------------- Simulation parameters -----------------------------
    "nx":[64, 64, 64],          	# Simulation grid numbers
    "lx":[64*0.11, 64*0.11, 64*0.11],   # Simulation box size as a_Ref * N_Ref^(1/2) unit,
                                # where "a_Ref" is reference statistical segment length
                                # and "N_Ref" is the number of segments of reference linear homopolymer chain.

    "chain_model":"discrete",   # "discrete" or "continuous" chain model
    "ds":1/90,                  # Contour step interval, which is equal to 1/N_Ref.
    "chi_n":25.0,               # Interaction parameter, Flory-Huggins params * N

    "segment_lengths":{         # Relative statistical segment length compared to "a_Ref.
        "A":np.sqrt(eps*eps/(eps*eps*f + (1-f))), 
        "B":np.sqrt(    1.0/(eps*eps*f + (1-f))), },

    "distinct_polymers":[{      # Distinct Polymers
        "volume_fraction":0.8,  # volume fraction of polymer chain
        "blocks":[              # AB diBlock Copolymer
            {"type":"A", "length":f, }, 	# A-block
            {"type":"B", "length":2*(1-f)}, 	# B-block
            {"type":"A", "length":f}, 		# A-block
        ],},
	{
        "volume_fraction":0.2,  
        "blocks":[              # Random Copolymer
            {"type":"random", "length":0.5, "fraction":{"A":0.5, "B":0.5}, },
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

# Set initial fields
print("w_minus and w_plus are initialized to random phase")
w_plus  = np.random.normal(0.0, 1.0, np.prod(params['nx'])),
w_minus = np.random.normal(0.0, 1.0, np.prod(params['nx'])),

# Initialize calculation
simulation = deep_langevin_fts.DeepLangevinFTS(params=params)

# Make training data
simulation.make_training_data(w_minus=w_minus, w_plus=w_plus)

# Train model
simulation.train_model()

# Find best epoch
input_fields_data = loadmat("LastTrainingStep.mat", squeeze_me=True)
simulation.find_best_epoch(w_minus=input_fields_data["w_minus"], w_plus=input_fields_data["w_plus"])

# Run
simulation.run(w_minus=input_fields_data["w_minus"], w_plus=input_fields_data["w_plus"],
   max_step=params["langevin"]["max_step"], model_file="best_epoch.pth")
