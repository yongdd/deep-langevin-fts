import os
import time
import random
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

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[10,20,30], gamma=0.5,
            verbose=False)
        return [optimizer], [scheduler]
    
    def on_train_start(self):
        total_params = sum(p.numel() for p in self.parameters())
        self.log('total_params', float(total_params))
    
    def on_epoch_start(self):
        self.log('learning_rate', self.optimizers().param_groups[0]['lr'])

    def on_epoch_end(self):
        torch.save(self.state_dict(), 'saved_model_%d.pth' % (self.current_epoch) )

    def NRMSLoss(self, target, output):
        return torch.sqrt(torch.mean((target - output)**2)/torch.mean(target**2))
        
    def training_step(self, train_batch, batch_idx):
        x = train_batch['input']
        y = train_batch['target']
        x = self(x)   
        loss = self.NRMSLoss(y, x)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x = val_batch['input']
        y = val_batch['target']
        x = self(x) 
        loss = self.NRMSLoss(y, x)
        self.log('val_loss', loss)

def train(model, data_dir):

    batch_size = 8
    num_workers = 4
    # training and validation data
    train_dir = os.path.join(data_dir, 'train')
    train_dataset = FtsDataset(train_dir)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
    print(len(train_dataset))
    
    #val_dir = os.path.join(data_dir, 'val')
    #val_dataset = FtsDataset(val_dir)    
    #val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)
    #print(len(val_dataset))
    
    # training
    trainer = pl.Trainer(
            gpus=7, num_nodes=1, max_epochs=50,
            precision=16,
            strategy=DDPPlugin(find_unused_parameters=False),
            benchmark=True, log_every_n_steps=5)
            
    trainer.fit(model, train_loader, None)

if __name__=="__main__":

    os.environ["PL_TORCH_DISTRIBUTED_BACKEND"]="gloo" #nccl or gloo

    data_dir = "data3d_gyroid"
    sample_file_path = "data3d_gyroid/train/"
         
    model_file = "trained_model.pth"
    model = TrainerAndModel()
    #model.load_state_dict(torch.load(model_file), strict=True)
    train(model, data_dir)
    deepfts = DeepFts(model)
    
    file_list = glob.glob(sample_file_path + "/*.npz")
    random.shuffle(file_list)
    
    for i in range(0,10):
        
        sample_file_name = file_list[i]
        sample_file_name_base = os.path.basename(sample_file_name).split('.')[0]
        sample_data = np.load(sample_file_name)
        nx = sample_data["nx"]
        lx = sample_data["lx"]

        wm = sample_data["w_minus"]
        wp = sample_data["w_plus"]
        gp = sample_data["g_plus"]
        wpd  = sample_data["w_plus_diff"]
        wpd_gen = deepfts.generate_w_plus(wm, gp, nx)
        X = np.linspace(0, lx[0], nx[0], endpoint=False)
        
        #print(np.mean(wpd), np.mean(wpd_gen))
        #wpd -= np.mean(wpd)
        #wpd_gen -= np.mean(wpd_gen)
        #print(np.mean(wpd), np.mean(wpd_gen))
    
        fig, axes = plt.subplots(2,2, figsize=(20,15))
    
        axes[0,0].plot(X, wm  [:nx[0]], )
        axes[0,1].plot(X, wp  [:nx[0]], )
        axes[1,0].plot(X, gp  [:nx[0]], )
        axes[1,1].plot(X, wpd [:nx[0]], )
        axes[1,1].plot(X, wpd_gen[:nx[0]], )

        plt.subplots_adjust(left=0.2,bottom=0.2,
                            top=0.8,right=0.8,
                            wspace=0.2, hspace=0.2)
        plt.savefig('%s.png' % (os.path.basename(sample_file_name_base)))
        plt.close()
