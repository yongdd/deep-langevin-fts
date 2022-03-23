import os
import numpy as np
import yaml
from scipy.io import *
from langevinfts import *
from deep_langevin_fts import *

#os.environ["CUDA_VISIBLE_DEVICES"]= "1"

# -------------- read parameters --------------
with open('input_parameters.yaml') as f:
    input_params = yaml.load(f, Loader=yaml.FullLoader)
    
input_data = np.load("GyroidScftInput.npz")

# -------------- langevin fts --------------
deepfts = DeepLangevinFTS(input_params)

# np.random.seed(5489)
deepfts.make_training_data(
    w_plus               = (input_data["w"][0] + input_data["w"][1])/2,
    w_minus              = (input_data["w"][0] - input_data["w"][1])/2,
    saddle_max_iter      = input_params['saddle']['max_iter'],
    saddle_tolerance     = float(input_params['saddle']['tolerance']),
    saddle_tolerance_ref = float(input_params['saddle']['tolerance_ref']),
    dt                   = input_params['langevin']['dt'],
    nbar                 = input_params['langevin']['nbar'],
    max_step             = input_params['training_data']['max_step'],
    path_dir             = input_params['training_data']['dir'],
    recording_period     = input_params['training_data']['recording_period'],
    recording_n_data     = input_params['training_data']['recording_n_data'])
