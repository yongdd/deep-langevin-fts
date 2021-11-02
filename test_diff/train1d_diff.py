import os
import sys
import logging
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset1d_diff import *
from model_atrnet1d_diff import *

class DeepFts1dDiff:
    def __init__(self, load_net=None):

        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f'Using device {self.device}')
        logging.info(f'Current cuda device {torch.cuda.current_device()}')
        #logging.info(f'Count of using GPUs {torch.cuda.device_count()}')
        
        self.train_folder_name = "./data1D_64_diff/train"
        self.test_folder_name = "./data1D_64_diff/eval"
        
        self.net = AtrNet1dDiff()
        if load_net:
            self.net.load_state_dict(torch.load(load_net, map_location=self.device))
            #self.net = torch.load(load_net, map_location=self.device)
            logging.info(f'Model loaded from {load_net}')

        #if torch.cuda.device_count() > 1:
        #    print("Let's use", torch.cuda.device_count(), "GPUs!")
        #    self.net = torch.nn.DataParallel(self.net)

        self.net.to(device=self.device)

    def generate_w_plus(self, w_minus, w_plus_gen, g_plus, nx):
        data = np.zeros([1, 3, nx[0]])
        
        data[0,0,:] = np.reshape(w_minus/10.0, (1, 1, nx[0]))
        data[0,1,:] = np.reshape(w_plus_gen/10.0, (1, 1, nx[0]))
        data[0,2,:] = np.reshape(g_plus*10.0, (1, 1, nx[0]))
        data = torch.tensor(data, dtype=torch.float32).to(self.device)
        #print(type(data), data.shape)
        with torch.no_grad():
            w_plus_diff = self.net(data).cpu().numpy()/10.0
            return np.reshape(w_plus_diff.astype(np.float64), nx[0])
            
    def eval_net(self, net, loader, device, criterion, writer, global_step):
        net.eval()
        n_val = len(loader)
        loss = 0

        with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
            for batch in loader:        
                data = batch["data"].to(device)
                target = batch["target"].to(device)  
                with torch.no_grad():
                    y_pred = net(data)
                loss += criterion(y_pred, target).item()
                pbar.update(data.shape[0])

            #writer.add_images('exchange', data[0,0,:], global_step, dataformats="HW")
            #writer.add_images('pred', y_pred[0,0,:], global_step, dataformats="HW")
            #writer.add_images('true', target[0,0,:], global_step, dataformats="HW")

        return loss / n_val

    def train_net(self,):
        
        net = self.net
        device = self.device
        total_params = sum(p.numel() for p in net.parameters())
        
        lr = 1e-4
        epochs = 100
        batch_size = 128
        log_dir = "logs"
        output_dir = "checkpoints"
                
        train = FtsDataset1dDiff(self.train_folder_name)
        val = FtsDataset1dDiff(self.test_folder_name)
        
        n_train = len(train)
        n_val = len(val)
        
        train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=False)
        val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=False, drop_last=False) 
           
        writer = SummaryWriter(log_dir=log_dir, comment=f'LR_{lr}_BS_{batch_size}')

        global_step = 0
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20,30,100,1000], gamma=0.5, verbose=True)
        logging.info(f'''Starting training:
            Epochs:          {epochs}
            Batch size:      {batch_size}
            Learning rate:   {lr}
            Training size:   {n_train}
            Validation size: {n_val}
            Checkpoints:     {output_dir}
            Device:          {device.type}
            Optimizer        {optimizer.__class__.__name__}
            Criterion        {criterion.__class__.__name__}
            Total Params     {total_params}
        ''')
        
        writer.add_scalar('total_params', total_params)
        
        global_step = 0
        for epoch in range(epochs):
            net.train()
            epoch_loss = 0
            with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='data') as pbar:
                for batch in train_loader:                
                    data = batch["data"].to(device)
                    target = batch["target"].to(device)              
                    y_pred = net(data)
                    loss = criterion(y_pred, target)
                    epoch_loss += loss.item()
                    
                    optimizer.zero_grad()
                    loss.backward()
                    #nn.utils.clip_grad_value_(net.parameters(), 0.1)
                    optimizer.step()
            
                    writer.add_scalar('Loss/train', loss.item(), global_step)
                    pbar.set_postfix(**{'loss (batch)': loss.item()})
                    pbar.update(data.shape[0])
                    
                    global_step += 1
            for tag, value in net.named_parameters():
                tag = tag.replace('.', '/')
                writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
            val_loss = self.eval_net(net, val_loader, device, criterion, writer, global_step)
            logging.info('Validation loss: {}'.format(val_loss))
            writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)
            writer.add_scalar('Loss/test', val_loss, global_step)
            try:
                os.mkdir(output_dir)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            scheduler.step()
            torch.save(net.state_dict(), os.path.join(output_dir, f'CP_epoch{epoch + 1}.pth'))
            #torch.save(net, os.path.join(output_dir, f'CP_epoch{epoch + 1}.pth'))
            logging.info(f'Checkpoint {epoch + 1} saved !')
        #torch.save(net, os.path.join(output_dir, f'epoch{epoch + 1}.pth'))
        torch.cuda.empty_cache()

if __name__ == '__main__':
    
    #os.environ["CUDA_VISIBLE_DEVICES"]= "1"
    model = DeepFts1d()
    #model = DeepFts1d("checkpoints/CP_epoch50.pth")
    model.train_net()
    
    sample_file_name = "./data1D_64_diff/eval/fields_300000.npz"
    sample_data = np.load(sample_file_name)
    nx = sample_data["nx"]

    X0 = np.reshape(sample_data["w_minus"], (1, 1, nx[0]))
    X1 = np.reshape(sample_data["w_plus_gen"],  (1, 1, nx[0]))
    X2 = np.reshape(sample_data["g_plus"],  (1, 1, nx[0]))
    Y     = np.reshape(sample_data["w_plus_gen"] - sample_data["w_plus"],  (1, 1, nx[0]))*10
    Y_gen = np.reshape(model.generate_w_plus(X0, X1, X2, nx)*10, (1, 1, nx[0]))
    vmin = np.min([np.min(Y), np.min(Y_gen)])
    vmax = np.max([np.max(Y), np.max(Y_gen)])
    
    fig, axes = plt.subplots(2,2, figsize=(20,20))
    
    axes[0,0].plot(X1[0,0,:])
    axes[1,0].plot(Y    [0,0,:])
    axes[1,0].plot(Y_gen[0,0,:])
    axes[1,1].plot(Y[0,0,:]-Y_gen[0,0,:])
     
    plt.subplots_adjust(left=0.2,bottom=0.2,
                        top=0.8,right=0.8,
                        wspace=0.2, hspace=0.2)
    plt.savefig('w_plus_minus.png')
