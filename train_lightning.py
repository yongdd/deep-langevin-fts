import os
import time
import torch
from torch.utils.data import DataLoader
from pytorch_lightning.plugins import DDPPlugin
from dataset import *
from pl_atrnet import *
from pl_aspp import *

class DeepFts():
    def __init__(self, dim, load_net=None):
        # model
        self.model = LitAtrNet(dim=dim, in_channels=3, mid_channels = 64, out_channels=1, kernel_size = 3)
        #self.model = LitAsppNet(dim=dim, in_channels=3, mid_channels = 64, out_channels=1, kernel_size = 3) 
        if load_net:
            self.model.load_state_dict(torch.load(load_net), strict=True)
        self.model.cuda()
        
    def generate_w_plus(self, w_minus, g_plus, nx):
        
        self.model.eval()
        data = np.zeros([1, 3, np.prod(nx)])
        data[0,0,:] = w_minus/10.0
        data[0,1,:] = g_plus
        normal_factor = np.std(data[0,1,:])
        data[0,1,:] /= normal_factor
        data[0,2,:] = 0.0 #np.log10(normal_factor)
        
        data = torch.tensor(np.reshape(data, [1, 3] + list(nx)), dtype=torch.float32).cuda()
        with torch.no_grad():
            output = self.model(data).detach().cpu().numpy()
            w_plus = np.reshape(output.astype(np.float64)*normal_factor, np.prod(nx))
            return w_plus

    def train(self,):
        os.environ["PL_TORCH_DISTRIBUTED_BACKEND"]="gloo" #nccl or gloo

        # data
        data_dir = "/hdd/hdd2/yong/L_FTS/2d_periodic/data2d_64_wp_diff"
        train_dir = os.path.join(data_dir, 'train')
        val_dir = os.path.join(data_dir, 'val')

        train_dataset = FtsDataset(train_dir)
        val_dataset = FtsDataset(val_dir)

        train_loader = DataLoader(train_dataset, batch_size=256, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=256, num_workers=4)

        # training
        trainer = pl.Trainer(
                gpus=2, num_nodes=1, max_epochs=10,
                precision=32, 
                strategy=DDPPlugin(find_unused_parameters=False),
                #profiler="simple",
                #limit_train_batches=0.5
                #default_root_dir="checkpoints"
                )
        trainer.fit(self.model, train_loader, val_loader)

if __name__=="__main__":
    deepfts = DeepFts(dim=2)
    deepfts.train()
