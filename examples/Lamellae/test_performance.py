import os
import sys
import numpy as np
import yaml
from scipy.io import *
from langevinfts import *
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from inference_net import *
from deep_langevin_fts import *

#os.environ["CUDA_VISIBLE_DEVICES"]= "1"

# -------------- read parameters --------------
with open('lamella_input_parameters.yaml') as f:
    input_params = yaml.load(f, Loader=yaml.FullLoader)

# -------------- override parameters --------------
# Load Data
input_data = loadmat("lamella_equil_chin17.0.mat", squeeze_me=True)

# Simulation Box
input_params['nx'] = input_data['nx'].tolist() 
input_params['lx'] = input_data['lx'].tolist()

# Polymer Chain
input_params['chain']['n_segment'] = input_data['N']
input_params['chain']['f']         = input_data['f']
input_params['chain']['chi_n']     = input_data['chi_n']
input_params['chain']['model']     = input_data['chain_model']

# -------------- deep learning --------------
use_deep_learning = True
model_file = "lamella_atr_cas_mish_32.pth"
print(model_file)

torch.set_num_threads(1)
if (use_deep_learning):
    net = InferenceNet(dim=3, features=32)
    net.load_state_dict(torch.load(model_file), strict=True)
else:
    net = None
    
# -------------- langevin fts --------------
deepfts = DeepLangevinFTS(input_params)

# np.random.seed(5489)
(total_saddle_iter, saddle_iter_per, time_duration_per,
time_pseudo_ratio, time_neural_net_ratio, total_net_failed, total_error_level) \
    = deepfts.run(
        w_plus           = input_data["w_plus"],
        w_minus          = input_data["w_minus"],
        saddle_max_iter  = input_params['saddle']['max_iter'],
        saddle_tolerance = float(input_params['saddle']['tolerance']),
        dt               = input_params['langevin']['dt'],
        nbar             = input_params['langevin']['nbar'],
        max_step         = 100,
        net              = net)
        
# estimate execution time
print( "Total iterations for saddle points: %d, iter per step: %f" %
    (total_saddle_iter, saddle_iter_per))
print( "Total time: %f, time per step: %f" %
    (time_duration_per*total_saddle_iter, time_duration_per) )
print( "Pseudo time ratio: %f, deep learning time ratio: %f" %
    (time_pseudo_ratio, time_neural_net_ratio) )
print( "The number of times that the neural-net could not reduce the incompressible error and switched to Anderson mixing: %d times" % (total_net_failed) )
