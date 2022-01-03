import os
import time
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from pytorch_lightning.plugins import DDPPlugin
from dataset import *
from model.model_pl_atrnet import *
from model.model_pl_atrxnet import *
from model.model_pl_asppnet import *
from model.model_pl_gcnet import *
from model.model_pl_unet import *
from model.model_pl_sqnet import *
from model.model_pl_resnet import *

class TrainerAndModel(LitAtrNet): # LitUNet2d, LitAtrNet, LitAsppNet, LitAtrXNet, LitGCNet, LitSqNet, LitResNet
    def __init__(self, dim):
        super().__init__(dim)
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[10,30], gamma=0.5,
            verbose=False)
        return [optimizer], [scheduler]
    
    def on_train_start(self):
        total_params = sum(p.numel() for p in self.parameters())
        self.log('total_params', float(total_params))
    
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
        
        # # set the qconfig for PTQ
        # qconfig = torch.quantization.get_default_qconfig('fbgemm')
        # # or, set the qconfig for QAT
        # qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        # # set the qengine to control weight packing
        # torch.backends.quantized.engine = 'fbgemm'
        # # set quantization config for server (x86)
        # deploymentmyModel.qconfig = torch.quantization.get_default_qconfig('fbgemm')

        # # insert observers
        # torch.quantization.prepare(self.model, inplace=True)
        # # Calibrate the model and collect statistics

        # # convert to quantized version
        # torch.quantization.convert(self.model, inplace=True)
        
        self.model.half().cuda()
        #self.model.cuda()
        
    def generate_w_plus(self, w_minus, g_plus, nx):
        
        normal_factor = 10.0 # an arbitrary normalization factor for rescaling
        self.model.eval()
        X = np.zeros([1, 3, np.prod(nx)])
        X[0,0,:] = w_minus/normal_factor 
        X[0,1,:] = g_plus
        std_g_plus = np.std(X[0,1,:])
        X[0,1,:] /= std_g_plus
        X[0,2,:] = np.log(std_g_plus)
        
        X = torch.tensor(np.reshape(X, [1, 3] + list(nx)), dtype=torch.float16).cuda()
        with torch.no_grad():
            output = self.model(X).detach().cpu().numpy()
            w_plus = np.reshape(output.astype(np.float64)*std_g_plus*normal_factor, np.prod(nx))
            return w_plus

    def train(self, data_dir):
        os.environ["PL_TORCH_DISTRIBUTED_BACKEND"]="gloo" #nccl or gloo

        # data
        train_dir = os.path.join(data_dir, 'train')
        val_dir = os.path.join(data_dir, 'val')

        train_dataset = FtsDataset(train_dir)
        val_dataset = FtsDataset(val_dir)

        print(len(train_dataset), len(val_dataset))

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
                benchmark=True, log_every_n_steps=10
                )
        trainer.fit(self.model, train_loader, val_loader)

if __name__=="__main__":

    dim = 3
    if (dim == 1):
        data_dir = "data1d_64"
        sample_file_path = "data1d_64/val/"
    elif (dim == 2):
        data_dir = "data2d_64"
        sample_file_path = "data2d_64/val/"
    elif (dim == 3):
        data_dir = "data3d_gyroid_dis_4"
        sample_file_path = "data3d_gyroid_dis_4/val/"
         
    model_file = "saved_model_49.pth"
    deepfts = DeepFts(dim=dim, load_net=model_file)
    #deepfts = DeepFts(dim=dim)
    #deepfts.train(data_dir)
    deepfts.half_cuda()


    #fields_1835_000624_032
    #fields_1835_000732_000
    if dim == 1 or dim == 2:
        langevin_iter = 400000 # 40000
    elif dim == 3:
        langevin_iter = 732
    saddle_iter = 0
    sample_file_name = sample_file_path + "fields_1835_%06d_%03d.npz" % (langevin_iter, saddle_iter)
    sample_data = np.load(sample_file_name)
    nx = sample_data["nx"]
    lx = sample_data["lx"]

    wm = sample_data["w_minus"]
    wp = sample_data["w_plus"]
    gp = sample_data["g_plus"]
    wpd  = sample_data["w_plus_diff"]
    wpd_gen = deepfts.generate_w_plus(wm, gp, nx)
    X = np.linspace(0, lx[0], nx[0], endpoint=False)
    print(np.mean(wpd), np.mean(wpd_gen))
    
    #with torchprof.Profile(deepfts.model, use_cuda=True, profile_memory=True) as prof:
    #    wpd_gen = deepfts.generate_w_plus(wm, gp, nx)
    #print(prof.display(show_events=False))
    
    #normal_factor =  np.std(gp)
    #gp /= normal_factor
    #wpd /= normal_factor*10
    
    fig, axes = plt.subplots(2,2, figsize=(20,15))
    
    axes[0,0].plot(X, wm  [:nx[0]], )
    axes[0,1].plot(X, wp  [:nx[0]], )
    axes[1,0].plot(X, gp  [:nx[0]], )
    axes[1,1].plot(X, wpd [:nx[0]], )
    axes[1,1].plot(X, wpd_gen[:nx[0]], )

    #axes[1,0].set_ylim([-0.3, 0.4])
    #axes[1,1].set_ylim([-2.5, 2.5])
    #axes[1,0].set_ylim([-2.5, 2.5])
    #axes[1,1].set_ylim([-10, 10])

    plt.subplots_adjust(left=0.2,bottom=0.2,
                        top=0.8,right=0.8,
                        wspace=0.2, hspace=0.2)
    plt.savefig('w_plus_minus_%06d_%03d.png' % (langevin_iter, saddle_iter))
