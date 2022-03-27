# Deep Langevin FTS
Langevin Field-Theoretic Simulation (L-FTS) Accelerated by Deep Learning

# Features
* Diblock Copolymer Melt
* Periodic Boundaries  
* Accelerating L-FTS using Deep Learning

# Dependencies

#### 1. Anaconda

#### 2. Langevin FTS
  L-FTS for Python   
  https://github.com/yongdd/langevin-fts

#### 3. PyTorch
  An open source machine learning framwork   
  https://pytorch.org/get-started/locally/

#### 4. PyTorch-lightning
  High-level interface for PyTorch   
  https://www.pytorchlightning.ai/

* * *
`Langevin FTS`, `PyTorch` and `PyTorch-lightning` should be installed in the same virtual environment. For instance, if you have installed `Langevin FTS` in virtual environment `envlfts`, install `PyTorch` and `PyTorch-lightning` after activating `envlfts`. Type following commands if the name of your virtual environment is `envlfts`.
   
  `conda activate envlfts`   
  `conda install pip matplotlib pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch`   
  `pip install pytorch-lightning`   
* * *   
You can run `python run_simulation.py` with pretrained model to test your installation.
  
# Usage

#### 1. Set Simulation Parameters
Edit `input_parameters.yaml`.  

#### 2. Make Training Data
Run `make_training_data.py`.   
    
  `python make_training_data.py`  

Training data will be stored in `data_training` folder. And you will get `LastTrainingData.mat` file. This can be used as inital field for `find_best_epoch.py` and `run_simulation.py`.   

#### 3. Train a Neural Network
If you are plan to use multiple GPUs for training, edit `gpus` in `train.py`. To obtain the same training results using multiple GPUs, you need to change `batch_size` so that `gpus` * `batch_size` does not change. For instance, if you use 4 GPUs, set `gpus=4` and `batch_size=2`, which is effectively the same as setting `gpus=1` and `batch_size=8`. Lastly, `find_best_epoch.py` will tell you which training result is the best.   
   
  `python train.py`   
  `python find_best_epoch.py`  
   
For each epoch, the weight of model will be stored in `saved_model_weights` folder. The training result is not always the same. If you are not satified with the result, run `train.py` once again.   

#### 4. Run Simulation
Edit `run_simulation.py` to use the best epoch, and run simulation.   
   
  `python run_simulation.py`  
   
Polymer density, fields and structure function will be recored in `data_simulation` folder.   

#### Data Visualization 
For visualization, Matlab and Python scripts are provided in `tools` folder.
