import numpy as np
import yaml
from scipy.io import *
import scipy.ndimage
from langevinfts import *
from saddle_net import *
from deep_langevin_fts import *

#os.environ["CUDA_VISIBLE_DEVICES"]= "2"

# -------------- read input parameters and data --------------
with open('input_parameters.yaml') as f:
    input_params = yaml.load(f, Loader=yaml.FullLoader)
input_data = loadmat("LastTrainingStep.mat", squeeze_me=True)

# Interpolate input data on input paramter grid
w_plus = scipy.ndimage.zoom(np.reshape(input_data["w_plus"], input_data["nx"]), input_params["nx"]/input_data["nx"])
w_minus = scipy.ndimage.zoom(np.reshape(input_data["w_minus"], input_data["nx"]), input_params["nx"]/input_data["nx"])

print(w_plus.shape)
print(w_minus.shape)

# -------------- deep learning --------------
use_deep_learning = True
model_file = "pretrained_models/gyroid_atr_cas_mish_32.pth"
print(model_file)

torch.set_num_threads(1)
if (use_deep_learning):
    net = SaddleNet(dim=3, features=32)
    net.load_state_dict(torch.load(model_file), strict=True)
else:
    net = None
    
# -------------- langevin fts --------------
deepfts = DeepLangevinFTS(input_params)

# np.random.seed(5489)
(total_saddle_iter, saddle_iter_per, time_duration_per,
time_pseudo_ratio, time_neural_net_ratio, total_net_failed, total_error_level) \
    = deepfts.run(
        w_plus              = np.reshape(w_plus, np.prod(input_params["nx"])),
        w_minus             = np.reshape(w_minus, np.prod(input_params["nx"])),
        saddle_max_iter     = input_params['saddle']['max_iter'],
        saddle_tolerance    = float(input_params['saddle']['tolerance']),
        dt                  = input_params['langevin']['dt'],
        nbar                = input_params['langevin']['nbar'],
        max_step            = input_params['simulation_data']['max_step'],
        path_dir            = input_params['simulation_data']['dir'],
        recording_period    = input_params['simulation_data']['recording_period'],
        sf_computing_period = input_params['simulation_data']['sf_computing_period'],
        sf_recording_period = input_params['simulation_data']['sf_recording_period'],
        net=net)
        
# estimate execution time
print( "Total iterations for saddle points: %d, iter per step: %f" %
    (total_saddle_iter, saddle_iter_per))
print( "Total time: %f, time per step: %f" %
    (time_duration_per*total_saddle_iter, time_duration_per) )
print( "Pseudo time ratio: %f, deep learning time ratio: %f" %
    (time_pseudo_ratio, time_neural_net_ratio) )
print( "The number of times that the neural-net could not reduce the incompressible error and switched to Anderson mixing: %d times" % (total_net_failed) )
