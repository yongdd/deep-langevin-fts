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
<<<<<<< HEAD
        self.loss = torch.nn.MSELoss()
=======
        self.milestones = [20]

    def set_milestones(self, milestones):
        self.milestones = milestones
>>>>>>> 7466615e18557dfe83e8b879515f782906fdf39b

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
<<<<<<< HEAD
            optimizer, milestones=[100], gamma=0.2,
=======
            optimizer, milestones=self.milestones, gamma=0.1,
>>>>>>> 7466615e18557dfe83e8b879515f782906fdf39b
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

    #def NRMSLoss(self, target, output):
        #return torch.sqrt(torch.mean((target - output)**2)/torch.mean(target**2))
        #return torch.sqrt(torch.mean((target - output)**2))
        #return torch.mean((target - output)**2)
        #return torch.mean(torch.abs(target - output))
      
    def training_step(self, train_batch, batch_idx):
        x = train_batch['input']
        y = train_batch['target']
        x = self(x)   
        loss = self.loss(y, x)
        self.log('train_loss', loss)
        return loss

if __name__=="__main__":

    os.environ["PL_TORCH_DISTRIBUTED_BACKEND"]="gloo" #nccl or gloo
    os.environ["CUDA_VISIBLE_DEVICES"]= "3,4,5,6"

    data_dir = "data_training"
<<<<<<< HEAD
    #model_file = "saved_model_weights/epoch_89.pth"
    #model_file = "pretrained_models/gyroid.pth"
=======
    #model_file = "1st.pth"
>>>>>>> 7466615e18557dfe83e8b879515f782906fdf39b
    model = TrainerAndModel()
    #model.load_state_dict(torch.load(model_file), strict=True)
    
    # training data    
    train_dataset = FtsDataset(data_dir)
    train_loader = DataLoader(train_dataset, batch_size=2, num_workers=4)
    print(len(train_dataset))
    
    # training
    trainer = pl.Trainer(
<<<<<<< HEAD
            gpus=4, num_nodes=1, max_epochs=200, precision=16,
            strategy=DDPPlugin(find_unused_parameters=False),
            benchmark=True, log_every_n_steps=5)
=======
            gpus=4, num_nodes=1, max_epochs=50, precision=16,
            strategy=DDPPlugin(find_unused_parameters=False),
            benchmark=True, log_every_n_steps=5)

>>>>>>> 7466615e18557dfe83e8b879515f782906fdf39b
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
       
        print(sample_file_name)

        target = wpd/np.std(gp)
        output = wpd_gen/np.std(gp)
        loss = np.sqrt(np.mean((target - output)**2)/np.mean(target**2))
        #print(np.std(target, dtype=np.float64),
        #      np.std(output, dtype=np.float64),
        #      np.sqrt(np.mean((target-output)*(target-output), dtype=np.float64))/
        #      np.mean(target*target, dtype=np.float64) )

        print(np.std(wm, dtype=np.float64),
              np.std(gp, dtype=np.float64),
              np.std(target, dtype=np.float64),
              np.std(output, dtype=np.float64),
              np.sqrt(np.mean((target-output)*(target-output), dtype=np.float64)),
              np.mean(np.abs(target-output), dtype=np.float64))
        #target = torch.tensor(wpd/np.std(gp))
        #output = torch.tensor(wpd_gen/np.std(gp))
        #loss = torch.sqrt(torch.mean((target - output)**2)/torch.mean(target**2))
        #print(torch.std(target, dtype=np.float64),
        #      torch.std(output, dtype=np.float64),
        #      torch.mean(target*target, dtype=np.float64) )
