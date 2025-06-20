import os
import time
import numpy as np
from scipy.io import savemat, loadmat
from scipy.ndimage import gaussian_filter
import scipy.ndimage
import deep_langevin_fts


f = 0.4        # A-fraction of major BCP chain, f
eps = 1.0      # a_A/a_B, conformational asymmetry

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

    "chi_n": {"A,B":18.0},     # Bare interaction parameter, Flory-Huggins params * N_Ref

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
        "recording_period":20000,       # Period for recording concentrations and fields
        "sf_computing_period":10,       # Period for computing structure function
        "sf_recording_period":100000,   # Period for recording structure function
    },

    "saddle":{                # Iteration for the pressure field 
        "max_iter" :400,      # Maximum number of iterations
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

    "wtmd":{                        # Well-tempered metadynamics
        "l":4,                      # ℓ-norm
        "kc":6.02,                  # screening out frequency
        "dT":5.0,                   # delta T/T
        "sigma_psi":0.16,           # σ_Ψ
        "psi_min":0.0,              # Ψ_min
        "psi_max":10.0,             # Ψ_max
        "dpsi":1e-3,                # dΨ, bin width of u, up, I0, I1
        "update_freq":1000,         # Update frequency
        "recording_period":100000,  # Period for recording statistics
    },

    "verbose_level":1,      # 1 : Print at each Langevin step.
                            # 2 : Print at each saddle point iteration.

    #---------------- Training parameters -----------------------------
    
    "training":{           
        # Training Data         
        "data_dir":"data_training", # Directory name
        "max_step":10000,           # Langevin steps for collecting training data
        "recording_period":5,       # Make training data every 5 Langevin steps
        "recording_n_data":4,       # Make 4 training data
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
        "features":32,                      # The number of features for each convolution layer

        # Data Loader
        "batch_size":8,                      # Batch size
        "num_workers":8,                     # The number of workers for data loading
    },
}
# Set random seed
# If you want to obtain different results for each execution, set random_seed=None
random_seed = 12345

# Set initial fields
input_scft_fields = loadmat("GyroidInput.mat", squeeze_me=True)
w_A = input_scft_fields["w_A"]
w_B = input_scft_fields["w_B"]
#w_A = np.random.normal(0.0, 1.0, np.prod(params['nx'])),
#w_B = np.random.normal(0.0, 1.0, np.prod(params['nx'])),

initial_fields={"A": w_A, "B": w_B}

# # Interpolate input data on params["nx"], if necessary
# w_A = scipy.ndimage.zoom(np.reshape(w_A, input_scft_fields["nx"]), params["nx"]/input_scft_fields["nx"])
# w_B = scipy.ndimage.zoom(np.reshape(w_B, input_scft_fields["nx"]), params["nx"]/input_scft_fields["nx"])

# Initialize simulation
simulation = deep_langevin_fts.DeepLangevinFTS(params=params, random_seed=random_seed)

# Generate training data
# After training data are generated, the field configurations of the last Langevin step will be saved with the file name "LastTrainingLangevinStep.mat".
simulation.make_training_dataset(initial_fields=initial_fields, final_fields_configuration_file_name="LastTrainingLangevinStep.mat")

# Train model
simulation.train_model()

## Continue Training
#simulation.train_model(model_file="saved_model_weights/epoch_73.pth", epoch_offset=73)

# Find best epoch
# The best neural network weights will be saved with the file name "best_epoch.pth".
input_fields_data = loadmat("LastTrainingLangevinStep.mat", squeeze_me=True)
w_A = input_fields_data["w_A"]
w_B = input_fields_data["w_B"]
initial_fields={"A": w_A, "B": w_B}
simulation.find_best_epoch(initial_fields=initial_fields, best_epoch_file_name="best_epoch.pth")

# Run
simulation.run(initial_fields=initial_fields, max_step=params["langevin"]["max_step"], model_file="best_epoch.pth")

# # Continue simulation with recorded field configurations and random state.
# simulation.continue_run(field_file_name="fields_010000.mat",
#    max_step=params["langevin"]["max_step"], model_file="best_epoch.pth")

# # Continue wtmd simulation
# simulation.continue_run("wtmd_fields_5000000.mat", max_step=10*10**6, final_fields_configuration_file_name="LastLangevinStep.mat", prefix="wtmd_", model_file="best_epoch.pth", use_wtmd=True, wtmd_file_name="wtmd_statistics_5000000.mat")
