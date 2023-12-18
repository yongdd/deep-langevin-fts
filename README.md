# Deep Langevin FTS
Langevin Field-Theoretic Simulation (L-FTS) Accelerated by Deep Learning (DL)

# Features
* L-FTS incorporated with DL
* Polymer Melts in Bulk
* Arbitrary Acyclic Branched Polymers
* Arbitrary Mixtures of Block Copolymers and Homopolymers (+ 1 Random Copolymer)
* Any number of monomer types (e.g., AB- and ABC-types)
* Conformational Asymmetry
* Chain Models: Continuous, Discrete
* Pseudo-spectral Method (4th-order Method for Continuous Chain)
* Well-tempered metadynamics
* Leimkuhler-Matthews Method for Updating Exchange Field
* Random Number Generator: PCG64
* Platform: CUDA only

# Dependencies

#### Linux System

#### Anaconda

#### Langevin FTS for Python 
  https://github.com/yongdd/langevin-fts

# Installation

`Langevin FTS`, `PyTorch` and `PyTorch-lightning` should be installed in the same virtual environment. For instance, if you have installed `Langevin FTS` in virtual environment `lfts`, install `PyTorch` and `PyTorch-lightning` after activating `lfts` using the following commands. (Assuming the name of your virtual environment is `lfts`)
```Shell
# Activate virtual environment  
conda activate lfts   
# Download DL-FTS   
git clone https://github.com/yongdd/deep-langevin-fts.git  
# Install pytorch  
conda install pytorch==2.0.1 pytorch-cuda=11.8 -c pytorch -c nvidia
# Install pytorch-lightning  
pip install pytorch-lightning==2.0.7
```
The above commands will install the following libraries.   
  
#### PyTorch
  An open source machine learning framework  
  https://pytorch.org/get-started/locally/

#### PyTorch-lightning
  High-level interface for PyTorch   
  https://www.pytorchlightning.ai/

* * *   
After the installation, you can run `python Gyroid.py` in `examples/Gyroid` folder, which performs an L-FTS with a pretrained model to test your installation. You can compare its performance with Anderson mixing by repeating simulation after setting `model_file=None` in `Gyroid.py`.  

# Usage
```Shell
python TrainAndRun.py
```
#### 1. Set Simulation Parameters
All the simulation and training parameters are stored in `params` of `TrainAndRun.py`. If you plan to use DL but you do not want to touch the details, only edit the upper part of `params`.

Two GPUs can be used to calculate the polymer concentrations. To use this option, set `os.environ["LFTS_NUM_GPUS"]="2"`. Unfortunately, simulation time does not always decrease. Check the performance first.  

If you plan to use multiple GPUs for training, edit `gpus`. To obtain the same training results using multiple GPUs, you need to change `batch_size` so that `gpus` * `batch_size` does not change. For example, if you use 4 GPUs, set `gpus=4` and `batch_size=8`, which is effectively the same as setting `gpus=1` and `batch_size=32`. For each epoch, the weight of model will be stored in `saved_model_weights` folder.  

#### 2. make_training_dataset(), train_model(), find_best_epoch()
Initial fields are currently for gyroid phase. You may need to change the initial fields by modifying `initial_fields` of `make_training_dataset()`. Training data will be stored in `data_training` folder, and it will generate `LastTrainingLangevinStep.mat` file. A sample `LastTrainingLangevinStep.mat` file already exists, and this file or the generated file will be used as initial field for `find_best_epoch()` and `run()`.

Lastly, `find_best_epoch()` will tell you which training result is the best, and it will copy the weights of best epoch as `best_epoch.pth`. A sample `best_epoch.pth` file already exists. The training result is not always the same. If you are not satisfied with the result, run `train_model()`, `find_best_epoch()` once again.  

#### 3. run()
This will use `best_epoch.pth` obtained at the previous step. You can use a pre-trained model in `examples` folder instead. For example, set `model_file="examples/Gyroid/gyroid_atr_cas_mish_32.pth"` when you want to run simulation for gyroid phase. For those who do not want to use DL, set `model_file=None`. Polymer concentrations, fields and structure function will be stored in `data_simulation` folder. To continue the run, invoke `continue_run()`, which is currently commented out in `TrainAndRun.py`.

# Notes
* To see training logs, type `tensorboard --logdir=./lightning_logs/ --port=[your port, e.g, 6006 or 8008]` in the command line. And, access to `http://localhost:[your port]` of the same server.
* Matlab and Python scripts for visualization and renormalization are provided in `tools` folder of `yongdd/langevin-fts` repository.  
* In `examples` folder, input fields obtained using SCFT, pre-trained model weights, and field configurations at equilibrium states for several known BCP morphologies are provided.  
* Currently, the best neural network model is `LitAtrousCascadeMish` of `model/model/atr_cas_mish.py`, and it is set as default model of `class TrainAndInference` in `deep_langevin_fts.py`.  
* To run `TrainAndRun.py`, `deep_langevin_fts.py` should exist in the same directory.  
* Depending on the simulation parameters, DL-FTS can diverge because of the Anderson mixing. In this case, Langevin white noise is regenerated. If the Anderson mixing diverges too often, reduce the Anderson mixing start_error in parameter set.
* If your training is not successful constantly (switching to Anderson mixing happens too often, more than once per 10 Langevin steps), try followings.
  * increase `features` to 64.
  * increase `recording_n_data` to 5.
  * increase `max_step` to 20000.
# References
#### Well-tempered Metadynamics
+ T. M. Beardsley and M. W. Matsen, Well-tempered metadynamics applied to field-theoretic simulations of diblock copolymer melts, *J. Chem. Phys.* **2022**, 157, 114902
#### Exchange Field Update
+ B. Vorselaars, Efficient Langevin and Monte Carlo sampling algorithms: the case of
field-theoretic simulations, *J. Chem. Phys.* **2023**, 158, 114117

# Citation
Daeseong Yong, and Jaeup U. Kim, Accelerating Langevin Field-theoretic Simulation of Polymers with Deep Learning, *Macromolecules* **2022**, 55, 6505  