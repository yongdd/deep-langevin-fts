import os
import time
import numpy as np
from scipy.io import savemat, loadmat
from scipy.ndimage.filters import gaussian_filter
import deep_langevin_fts

# Major Simulation params
f = 0.5          # A-fraction of major BCP chain, f
eps = 1.0        # a_A/a_B, conformational asymmetry
n_sc = 50        # the number of side chains
sc_alpha = 0.3   # N_sc/ N_bb
chi_n = 49.3     # Interaction parameter, Flory-Huggins params*N_total

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
    "nx":[64,64,64],        # Simulation grid numbers
    "lx":[7.76,7.76,7.76],  # Simulation box size as a_Ref * N_Ref^(1/2) unit,
                            # where "a_Ref" is reference statistical segment length
                            # and "N_Ref" is the number of segments of reference linear homopolymer chain.

    "chain_model":"discrete",     # "discrete" or "continuous" chain model
    "ds":1/100,                   # Contour step interval, which is equal to 1/N_Ref.
    "chi_n": chi_n/total_alpha,   # Interaction parameter, Flory-Huggins params*N_Ref
    
    "segment_lengths":{         # Relative statistical segment length compared to "a_Ref.
        "A":np.sqrt(eps*eps/(eps*eps*f + (1-f))), 
        "B":np.sqrt(    1.0/(eps*eps*f + (1-f))), },

    "distinct_polymers":[{      # Distinct Polymers
        "volume_fraction":1.0,  # volume fraction of polymer chain
        "blocks": blocks,
        },],
    
    "langevin":{                # Langevin Dynamics
        "max_step":500000,      # Langevin steps for simulation
        "dt":0.02,              # Langevin step interval, delta tau*N_Ref
        "nbar":10000,           # invariant polymerization index, nbar
    },
    
    "recording":{                       # Recording Simulation Data
        "dir":"data_simulation",        # Directory name
        "recording_period":1000,        # period for recording concentrations and fields
        "sf_computing_period":10,       # period for computing structure function
        "sf_recording_period":10000,    # period for recording structure function
    },

    "saddle":{                # Iteration for the pressure field 
        "max_iter" :400,      # maximum the number of iterations
        "tolerance":1e-4,     # tolerance of incompressibility 
    },

    "am":{
        "max_hist":20,              # Maximum number of history
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

    # 2. If your simulations do not work well constantly, try followings
    #   a) Adjust "dt" so that the number of iterations is smaller than 100.
    #   b) increase `features` to 64.
    #   c) increase `max_epochs` to 300.
    #   d) increase `recording_n_data` to 6.

    "training":{           
        # Training Data         
        "data_dir":"data_training", # Directory name
        "max_step":10000,           # Langevin steps for collecting training data
        "recording_period":5,       # make training data every 5 Langevin steps
        "recording_n_data":3,       # make 3 training data
        "tolerance":1e-7,           # tolerance of incompressibility for training data

        # Training GPUs
        "gpus":1,                   # the number of gpus per node
        "num_nodes":1,              # the number of gpu nodes

        # Training Parameters
        "lr":1e-3,                           # Learning rate
        "precision":16,                      # training precision, [64, 32, 16] = [double, single, mixed] precisions
        "max_epochs":200,                    # the number of epochs
        "model_dir":"saved_model_weights",   # Directory for saved_model_weights

        # Model Parameters
        "features": 32,                      # the number of parameters

        # Data Loader
        "batch_size":8,                      # Batch size
        "num_workers":8,                     # the number of workers for data loading
    },
}
## random seed for MT19937
np.random.seed(5489)

# Set initial fields
print("w_minus and w_plus are initialized to gyroid phase")
input_scft_fields = loadmat("BottleBrushLamella3D.mat", squeeze_me=True)
w_plus  = (input_scft_fields["w_a"] + input_scft_fields["w_b"])/2,
w_minus = (input_scft_fields["w_a"] - input_scft_fields["w_b"])/2,
#w_plus  = np.random.normal(0.0, 1.0, np.prod(input_params['nx'])),
#w_minus = np.random.normal(0.0, 1.0, np.prod(input_params['nx'])),

# Initialize calculation
simulation = deep_langevin_fts.DeepLangevinFTS(params=params)

# Make training data
# After training data are generated, the field configurations of the last langevin step will be saved with the file name "LastTrainingStep.mat".
simulation.make_training_data(w_minus=w_minus, w_plus=w_plus, last_training_step_file_name="LastTrainingStep.mat")

# Train model
simulation.train_model()

# Find best epoch
# The best neural network weights will be saved with the file name "best_epoch.pth".
input_fields_data = loadmat("LastTrainingStep.mat", squeeze_me=True)
simulation.find_best_epoch(w_minus=input_fields_data["w_minus"], w_plus=input_fields_data["w_plus"], best_epoch_file_name="best_epoch.pth")

# Run
simulation.run(w_minus=input_fields_data["w_minus"], w_plus=input_fields_data["w_plus"],
   max_step=params["langevin"]["max_step"], model_file="best_epoch.pth")