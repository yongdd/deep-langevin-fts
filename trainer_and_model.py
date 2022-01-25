import os
import time
import random
import pathlib
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from pytorch_lightning.plugins import DDPPlugin
from dataset import *
from model.atrnet import *
from model.atrxnet import *
from model.asppnet import *
from model.gcnet import *
from model.unet import *
from model.sqnet import *
from model.resnet import *
from deep_fts import *

class TrainerAndModel(LitAtrNet): # LitUNet2d, LitAtrNet, LitAsppNet, LitAtrXNet, LitGCNet, LitSqNet, LitResNet
    def __init__(self, dim=3):
        super().__init__(dim)
        self.milestones = [20]

    def set_milestones(self, milestones):
        self.milestones = milestones

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.milestones, gamma=0.1,
            verbose=False)
        #optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        #scheduler = torch.optim.lr_scheduler.CyclicLR(
        #    optimizer, base_lr=1e-3, max_lr=1e-2, 
        #    step_size_up=5, step_size_down=5, mode='triangular',
        #    verbose=False)
        return [optimizer], [scheduler]
    
    def on_train_start(self):
        total_params = sum(p.numel() for p in self.parameters())
        self.log('total_params', float(total_params))
    
    def on_epoch_start(self):
        self.log('learning_rate', self.optimizers().param_groups[0]['lr'])
        #print('\n')

    def on_epoch_end(self):
        path = "saved_model_weights"
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, 'epoch_%d.pth' % (self.current_epoch)))

    def NRMSLoss(self, target, output):
        return torch.sqrt(torch.mean((target - output)**2)/torch.mean(target**2))
        
    def training_step(self, train_batch, batch_idx):
        x = train_batch['input']
        y = train_batch['target']
        x = self(x)   
        loss = self.NRMSLoss(y, x)
        self.log('train_loss', loss)
        return loss

if __name__=="__main__":

    os.environ["PL_TORCH_DISTRIBUTED_BACKEND"]="gloo" #nccl or gloo

    data_dir = "data_training"
    #model_file = "1st.pth"
    model = TrainerAndModel()
    #model.load_state_dict(torch.load(model_file), strict=True)
    
    # training data    
    train_dataset = FtsDataset(data_dir)
    train_loader = DataLoader(train_dataset, batch_size=2, num_workers=4)
    print(len(train_dataset))
    
    # training
    trainer = pl.Trainer(
            gpus=4, num_nodes=1, max_epochs=50, precision=16,
            strategy=DDPPlugin(find_unused_parameters=False),
            benchmark=True, log_every_n_steps=5)

    trainer.fit(model, train_loader, None)
    deepfts = DeepFts(model)
    deepfts.eval_mode()
    
    file_list = glob.glob(data_dir + "/*.npz")
    random.shuffle(file_list)
    
    for i in range(0,10):
        
        sample_file_name = file_list[i]
        sample_file_name_base = os.path.basename(sample_file_name).split('.')[0]
        sample_data = np.load(sample_file_name)
        nx = sample_data["nx"]
        lx = sample_data["lx"]

        wm = sample_data["w_minus"]
        gp = sample_data["g_plus"]
        wpd  = sample_data["w_plus_diff"]
        wpd_gen = deepfts.generate_w_plus(wm, gp, nx)
        X = np.linspace(0, lx[0], nx[0], endpoint=False)
    
        fig, axes = plt.subplots(2,2, figsize=(20,15))
    
        axes[0,0].plot(X, wm  [:nx[0]], )
        axes[1,0].plot(X, gp  [:nx[0]], )
        axes[1,1].plot(X, wpd [:nx[0]], )
        axes[1,1].plot(X, wpd_gen[:nx[0]], )

        plt.subplots_adjust(left=0.2,bottom=0.2,
                            top=0.8,right=0.8,
                            wspace=0.2, hspace=0.2)
        plt.savefig('%s.png' % (os.path.basename(sample_file_name_base)))
        plt.close()
