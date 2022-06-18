import os
import glob
import random
import pathlib
import matplotlib.pyplot as plt
from saddle_net import *

torch.set_num_threads(1)
model_file = "pretrained_models/gyroid_atr_par_32.pth"
model = SaddleNet(dim=3, features=32)
model.load_state_dict(torch.load(model_file), strict=True)
file_list = glob.glob("data_training/*.npz")
random.shuffle(file_list)

for i in range(0,10):
    
    file_name = file_list[i]
    file_name_base = os.path.basename(file_name).split('.')[0]
    data = np.load(file_name)
    nx = data["nx"]
    lx = data["lx"]

    wm = data["w_minus"]
    gp = data["g_plus"]
    wpd  = data["w_plus_diff"]
    wpd_gen = model.predict_w_plus(wm, gp, nx)
    X = np.linspace(0, lx[0], nx[0], endpoint=False)

    fig, axes = plt.subplots(2,2, figsize=(20,15))

    axes[0,0].plot(X, wm  [:nx[0]], )
    axes[1,0].plot(X, gp  [:nx[0]], )
    axes[1,1].plot(X, wpd [:nx[0]], )
    axes[1,1].plot(X, wpd_gen[:nx[0]], )

    plt.subplots_adjust(left=0.2,bottom=0.2,
                        top=0.8,right=0.8,
                        wspace=0.2, hspace=0.2)
    plt.savefig('%s.png' % (os.path.basename(file_name_base)))
    plt.close()
   
    print(file_name)

    target = wpd/np.std(gp)
    output = wpd_gen/np.std(gp)
    loss = np.sqrt(np.mean((target - output)**2)/np.mean(target**2))

    print(np.std(wm, dtype=np.float64),
          np.std(gp, dtype=np.float64),
          np.std(target, dtype=np.float64),
          np.std(output, dtype=np.float64),
          np.sqrt(np.mean((target-output)*(target-output), dtype=np.float64)),
          np.mean(np.abs(target-output), dtype=np.float64))
