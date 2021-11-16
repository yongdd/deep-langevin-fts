import os
import time
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from pytorch_lightning.plugins import DDPPlugin
from dataset import *
from model_pl_atrnet import *
from model_pl_atrxnet import *
from model_pl_asppnet import *
from model_pl_gcnet import *
from model_pl_unet import *

class TrainerAndModel(LitAtrXNet): # LitUNet2d, LitAtrNet, LitAsppNet, LitAtrXNet, LitGCNet
    def __init__(self, dim):
        super().__init__(dim)
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[10,20,30,40], gamma=0.5,
            verbose=False)
        return [optimizer], [scheduler]
    
    def on_train_start(self):
        total_params = sum(p.numel() for p in self.parameters())
        self.log('total_params', total_params)
    
    def on_epoch_start(self):
        self.log('learning_rate', self.optimizers().param_groups[0]['lr'])
        #print('\n')
        
    def on_epoch_end(self):
        #for tag, value in self.named_parameters():
            #tag = tag.replace('.', '/')
            #self.add_histogram('weights/' + tag, value.data, self.current_epoch)
            #self.add_histogram('grads/' + tag, value.grad.data, self.current_epoch)
        torch.save(self.state_dict(), 'saved_model_%d.pth' % (self.current_epoch) )

    def NRMSLoss(self, target, output):
        return torch.sqrt(torch.mean((target - output)**2))/torch.std(target)

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

class DeepFts():
    def __init__(self, dim, load_net=None):
        self.dim = dim
        # model
        self.model = TrainerAndModel(dim=dim)
        if load_net:
            self.model.load_state_dict(torch.load(load_net), strict=True)
    
    def half_cuda(self):
        self.model.half().cuda()
        
    def generate_w_plus(self, w_minus, g_plus, nx):
        
        self.model.eval()
        X = np.zeros([1, 3, np.prod(nx)])
        X[0,0,:] = w_minus/10.0
        X[0,1,:] = g_plus
        normal_factor = np.std(X[0,1,:])
        X[0,1,:] /= normal_factor
        X[0,2,:] = np.sqrt(normal_factor)
        
        X = torch.tensor(np.reshape(X, [1, 3] + list(nx)), dtype=torch.float16).cuda()
        with torch.no_grad():
            output = self.model(X).detach().cpu().numpy()
            w_plus = np.reshape(output.astype(np.float64)*(normal_factor*10), np.prod(nx))
            return w_plus

    def train(self, data_dir):
        os.environ["PL_TORCH_DISTRIBUTED_BACKEND"]="gloo" #nccl or gloo

        # data
        train_dir = os.path.join(data_dir, 'train')
        val_dir = os.path.join(data_dir, 'val')

        train_dataset = FtsDataset(train_dir)
        val_dataset = FtsDataset(val_dir)

        if self.dim == 1 or self.dim ==2 :
            batch_size = 128
        elif self.dim == 3 :
            batch_size = 32
        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)

        # training
        trainer = pl.Trainer(
                gpus=7, num_nodes=1, max_epochs=50,
                precision=16, #strategy='dp'
                strategy=DDPPlugin(find_unused_parameters=False),
                #benchmark=True,
                )
        trainer.fit(self.model, train_loader, val_loader)

if __name__=="__main__":

    dim = 2
    if (dim == 1):
        data_dir = "data1d_64_wp_diff"
        sample_file_name = "data1d_64_wp_diff/val/"
    elif (dim == 2):
        data_dir = "data2d_64_wp_diff"
        sample_file_name = "data2d_64_wp_diff/val/"
    elif (dim == 3):
        data_dir = "data3d_64_wp_diff"
        sample_file_name = "data3d_64_wp_diff/val/"
         
    #model_file = "saved_model_79.pth"
    #deepfts = DeepFts(dim=dim, load_net=model_file)
    deepfts = DeepFts(dim=dim)
    deepfts.train(data_dir)
    deepfts.half_cuda()

    if dim == 1 or dim == 2:
        langevin_iter = 400000 # 40000
    elif dim == 3:
        langevin_iter = 40000
    saddle_iter = 0
    sample_file_name += "fields_1_%06d_%03d.npz" % (langevin_iter, saddle_iter)
    sample_data = np.load(sample_file_name)
    nx = sample_data["nx"]

    X0 = sample_data["w_minus"]
    X1 = sample_data["g_plus"]
    Y  = sample_data["w_plus_diff"]
    Y_gen = deepfts.generate_w_plus(X0, X1, nx)
    
    print(np.mean(Y), np.mean(Y_gen))
    #Y_gen -= np.mean(Y_gen)
    
    fig, axes = plt.subplots(2,2, figsize=(20,20))
    
    axes[0,0].plot(X0[:nx[0]])
    axes[0,1].plot(X1 [:nx[0]])
    axes[1,0].plot(Y [:nx[0]])
    axes[1,0].plot(Y_gen[:nx[0]])
    #axes[1,1].plot(Y[0,0,:]-Y_gen[0,0,:])
     
    plt.subplots_adjust(left=0.2,bottom=0.2,
                        top=0.8,right=0.8,
                        wspace=0.2, hspace=0.2)
    plt.savefig('w_plus_minus_%06d_%03d.png' % (langevin_iter, saddle_iter))
