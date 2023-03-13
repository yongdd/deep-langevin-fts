import os
import sys
import time
import numpy as np
from scipy.io import savemat, loadmat
from scipy.ndimage.filters import gaussian_filter
import scipy.ndimage
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import deep_langevin_fts

f = 0.4        # A-fraction of major BCP chain, f
eps = 1.0       # a_A/a_B, conformational asymmetry

params = {
    #---------------- Simulation parameters -----------------------------

    # Neural-net is trained in 64^3 grids.
    # To use the trained NN in different simulation box size, we change "nx" as well as "lx", fixing grid interval "dx".

    "nx":[67,67,67],        # Simulation grid numbers
    "lx":[7.15*67/64,
          7.15*67/64,
          7.15*67/64],
                            # Simulation box size as a_Ref * N_Ref^(1/2) unit,
                            # where "a_Ref" is reference statistical segment length
                            # and "N_Ref" is the number of segments of reference linear homopolymer chain.

    "chain_model":"discrete",      # "discrete" or "continuous" chain model
    "ds":1/100,                    # Contour step interval, which is equal to 1/N_Ref.
    "chi_n": 12.0,                 # Interaction parameter, Flory-Huggins params*N_Ref.
    
    "segment_lengths":{            # Relative statistical segment length compared to "a_Ref.
        "A":np.sqrt(eps*eps/(eps*eps*f + (1-f))), 
        "B":np.sqrt(    1.0/(eps*eps*f + (1-f))), },

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
        "dt":0.8,               # Langevin step interval, delta tau*N_Ref
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

    "am":{
        "max_hist":60,              # Maximum number of history
        "start_error":8e-1,         # When switch to AM from simple mixing
        "mix_min":0.1,              # Minimum mixing rate of simple mixing
        "mix_init":0.1,             # Initial mixing rate of simple mixing
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
        "recording_n_data":3,       # Make 3 training data
        "tolerance":1e-7,           # tolerance of incompressibility for training data

        # Training GPUs
        "gpus":1,                   # The number of gpus per node
        "num_nodes":1,              # The number of gpu nodes

        # Training Parameters
        "lr":1e-3,                           # Learning rate
        "precision":16,                      # Training precision, [64, 32, 16] = [double, single, mixed] precisions
        "max_epochs":200,                    # The number of epochs
        "model_dir":"saved_model_weights",   # Directory for saved_model_weights

        # Model Parameters
        "features": 32,                      # The number of parameters

        # Data Loader
        "batch_size":8,                      # Batch size
        "num_workers":8,                     # The number of workers for data loading
    },
}
## random seed for MT19937
np.random.seed(5489)

# Set initial fields
print("w_minus and w_plus are initialized to gyroid phase")
#input_scft_fields = loadmat("fields_200000.mat", squeeze_me=True)
#w_plus  = (input_scft_fields["w_a"] + input_scft_fields["w_b"])/2,
#w_minus = (input_scft_fields["w_a"] - input_scft_fields["w_b"])/2,
#w_plus  = np.random.normal(0.0, 1.0, np.prod(input_params['nx'])),
#w_minus = np.random.normal(0.0, 1.0, np.prod(input_params['nx'])),

# Initialize calculation
simulation = deep_langevin_fts.DeepLangevinFTS(params=params)

# Make training data
# After training data are generated, the field configurations of the last langevin step will be saved with the file name "LastTrainingStep.mat".
#simulation.make_training_data(w_minus=w_minus, w_plus=w_plus, last_training_step_file_name="LastTrainingStep.mat")

# Train model
#simulation.train_model()

# Find best epoch
# The best neural network weights will be saved with the file name "best_epoch.pth".
input_data = loadmat("fields_050000.mat", squeeze_me=True)

# # Interpolate input data on params["nx"], if necessary
w_plus = input_data["w_plus"]
w_minus = input_data["w_minus"]

w_plus = scipy.ndimage.zoom(np.reshape(w_plus, input_data["nx"]), params["nx"]/input_data["nx"])
w_minus = scipy.ndimage.zoom(np.reshape(w_minus, input_data["nx"]), params["nx"]/input_data["nx"])

#simulation.find_best_epoch(w_minus=input_fields_data["w_minus"], w_plus=input_fields_data["w_plus"], best_epoch_file_name="best_epoch.pth")

# Run
simulation.run(w_minus=w_minus, w_plus=w_plus,
   max_step=1000, model_file="chin13_dt02.pth")

# Recording first a few iteration results for debugging and refactoring

# ---------- model file : chin13_dt02.pth ----------
#        1    1.021E-14  [ 3.8617386E+07  ]     6.092586061   3.0787251E-05 
# iteration, mass error, total partitions, total energy, incompressibility error
# ---------- Run  ----------
# Langevin step:  1
#        6    2.220E-16  [ 2.2627508E+07  ]     5.495184070   5.9909445E-05 
# Langevin step:  2
#        7   -2.331E-15  [ 2.6949115E+07  ]     5.633722474   7.0564521E-05 
# Langevin step:  3
#        6   -3.442E-15  [ 3.0692059E+07  ]     5.741129336   9.5968846E-05 
# Langevin step:  4
#        7    2.220E-15  [ 3.4552472E+07  ]     5.823287324   7.5809209E-05 
# Langevin step:  5
#        7    8.438E-15  [ 3.8341738E+07  ]     5.888506738   6.0000116E-05 
