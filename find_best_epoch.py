import os
import numpy as np
import yaml
from scipy.io import *
from langevinfts import *
from saddle_net import *
from deep_langevin_fts import *

#os.environ["CUDA_VISIBLE_DEVICES"]= "1"

# -------------- read input parameters and data --------------
with open('input_parameters.yaml') as f:
    input_params = yaml.load(f, Loader=yaml.FullLoader)
input_data = loadmat("LastTrainingData.mat", squeeze_me=True)

# -------------- deep learning --------------
saved_weight_dir = "saved_model_weights"
torch.set_num_threads(1)
net = SaddleNet(dim=3, mid_channels=32)

#-------------- test roughly ------------
deepfts = DeepLangevinFTS(input_params)

list_saddle_iter_per = []
print("iteration, mass error, total_partition, energy_total, error_level")
for i in range(30,100):
    
    model_file = os.path.join(saved_weight_dir ,"epoch_%d.pth" % (i))
    net.load_state_dict(torch.load(model_file), strict=True)
    print("---------- model file  ----------")
    print(model_file)
    (_, saddle_iter_per, _, _, _, _) = deepfts.run(
        w_plus              = input_data["w_plus"].copy(),
        w_minus             = input_data["w_minus"].copy(),
        saddle_max_iter     = input_params['saddle']['max_iter'],
        saddle_tolerance    = float(input_params['saddle']['tolerance']),
        dt                  = input_params['langevin']['dt'],
        nbar                = input_params['langevin']['nbar'],
        max_step            = 5,
        net =net)
    list_saddle_iter_per.append([model_file, saddle_iter_per])
sorted_saddle_iter_per = sorted(list_saddle_iter_per, key=lambda l:l[1])

#-------------- test top-10 epochs ------------
list_saddle_iter_per = []
for data in sorted_saddle_iter_per[0:10]:
    
    model_file = data[0]
    net.load_state_dict(torch.load(model_file), strict=True)
    print("---------- model file  ----------")
    print(model_file)
    (_, saddle_iter_per, _, _, _, _) = deepfts.run(
        w_plus              = input_data["w_plus"].copy(),
        w_minus             = input_data["w_minus"].copy(),
        saddle_max_iter     = input_params['saddle']['max_iter'],
        saddle_tolerance    = float(input_params['saddle']['tolerance']),
        dt                  = input_params['langevin']['dt'],
        nbar                = input_params['langevin']['nbar'],
        max_step            = 100,
        net =net)
    list_saddle_iter_per.append([model_file, saddle_iter_per])

sorted_saddle_iter_per = sorted(list_saddle_iter_per, key=lambda l:l[1])
print(*sorted_saddle_iter_per, sep = "\n")
