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
`Langevin FTS`, `PyTorch` and `PyTorch-lightning` should be installed in the same virtual environment. For instance, if you have installed `Langevin FTS` in virtual environment `envlfts`, install `PyTorch` and `PyTorch-lightning` after activating `envlfts` by following commands.   
   
  `conda activate envlfts`   
  `conda install pip matplotlib pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch`   
  `pip install pytorch-lightning`   
  
# Usage

#### 1. Set Simulation Parameters
Edit "input_parameters.yaml"  

#### 2. Make Training Data and Train Model
Run python scripts in following order. If you are plan to use multiple GPUs for training, edit `gpus` in `train.py` file. To obtain the same results using multiple gpus, you need to change 'batch_size' so that gpus` times `batch_size` is constant. For instance, if you use 4 GPUs, set `gpus=4` and `batch_size=2`. Or change your learning rate. Result of `find_best_epoch.py` will show that which is the best epoch.  
  `python make_training_data.py`  
  `python train.py`  
  `python find_best_epoch.py`  
#### 3. Run Simulation
Edit `run_simulation.py` to use the best epoch, and run simulation.   
  `python run_simulation.py`  
