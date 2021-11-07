# Ref. https://github.com/pytorch/examples/blob/master/imagenet

import os
import sys
import time
import logging
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from argparse import Namespace

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from dataset import *
from model_atrnet import *

def main():
    #args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'

    args = Namespace(
        dim = 2,
        data_dir = "/hdd/hdd2/yong/L_FTS/2d_periodic/data2d_64_wp_diff",
        cp_dir = "checkpoints_2",
        log_dir = "logs",
        lr = 1.0e-4,
        batch_size = 128, 
        epochs = 50,
        workers = 2,
        gpu = None,
        resume = None, # checkpoints_2
        start_epoch = 0,
        world_size = 2, # n_proc
        rank = -1,      # proc_id
        distributed = True,
        #multiprocessing_distributed = False,
        )

    try:
        os.mkdir(args.cp_dir)
        print('Created checkpoint directory')
    except OSError:
        pass
    
    ngpus_per_node = torch.cuda.device_count()
    if args.distributed:
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        print("Run main_worker with parallel")
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        print("Run single main_worker")
        main_worker(0, 1, args)

def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    #print(gpu, ngpus_per_node, args)

    writer = SummaryWriter(log_dir=args.log_dir, comment=f'LR_{args.lr}_BS_{args.batch_size}')

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        dist.init_process_group(backend='gloo', init_method='env://',
                rank=args.gpu, world_size=args.world_size)
        #print(dist.get_rank(), dist.get_world_size()),  dist.get_backend())

    # Create model
    model = AtrNet(args.dim)
    if not torch.cuda.is_available():
        print("using CPU, this will be slow")
    elif args.distributed:
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)
        # When using a single GPU per process and per
        # DistributedDataParallel, we need to divide the batch size
        # ourselves based on the total number of GPUs we have
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    else:
        model = model.cuda()

    total_params = sum(p.numel() for p in model.parameters())
    print(f'total_params: {total_params}')
    writer.add_scalar('total_params', total_params)

    # define loss function (criterion), optimizer and scheduler
    criterion = torch.nn.MSELoss().cuda(args.gpu)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20,30,100], gamma=0.5, verbose=True)

    # Data loading code
    train_dir = os.path.join(args.data_dir, 'train')
    val_dir = os.path.join(args.data_dir, 'val')

    train_dataset = FtsDataset(train_dir)
    val_dataset = FtsDataset(val_dir)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False, 
            num_workers=args.workers, pin_memory=True)  

    # optionally resume from a checkpoint
    # if args.resume:
        # if os.path.isfile(args.resume):
            # print("=> loading checkpoint '{}'".format(args.resume))
            # checkpoint = torch.load(args.resume)
            # args.start_epoch = checkpoint['epoch']
            # best_acc1 = checkpoint['best_acc1']
            # model.load_state_dict(checkpoint['state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer'])
            # print("=> loaded checkpoint '{}' (epoch {})"
                  # .format(args.resume, checkpoint['epoch']))
        # else:
            # print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
    
    #if args.evaluate:
    #    validate(val_loader, model, criterion, args)
    #    return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
        
        # train for one epoch
        train(train_loader, model, criterion, optimizer, writer, epoch, args)

        # evaluate on validation set
        validate(val_loader, model, criterion, writer, epoch, args)

        # ## remember best acc@1 and save checkpoint
        # #is_best = acc1 > best_acc1
        # #best_acc1 = max(acc1, best_acc1)

        # if not args.parallel or (args.parallel
                # and args.rank % ngpus_per_node == 0):
            # save_checkpoint({
                # 'epoch': epoch + 1,
                # 'arch': args.arch,
                # 'state_dict': model.state_dict(),
                # #'best_acc1': best_acc1,
                # 'optimizer' : optimizer.state_dict(),
            # }, is_best)
            
        if not args.distributed :
            torch.save(model.state_dict(), os.path.join(args.cp_dir, f'CP_epoch{epoch + 1}.pth'))
            print(f'Checkpoint {epoch + 1} saved !')
        if (args.distributed and args.gpu == 0):
            torch.save(model.module.state_dict(), os.path.join(args.cp_dir, f'CP_epoch{epoch + 1}.pth'))
            print(f'Checkpoint {epoch + 1} saved !')
              
        scheduler.step()

    torch.cuda.empty_cache()

def train(train_loader, model, criterion, optimizer, writer, epoch, args):

    # switch to train mode
    model.train()

    #n_train = len(train_loader.dataset)
    print(len(train_loader.dataset))
    print(len(train_loader))
    
    train_loss = 0.0
    n_train = 0
    with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{args.epochs}', unit=' batch') as pbar:
        for batch in train_loader:
            
            # data load
            data = batch["data"]
            target = batch["target"]  
            n_train += len(data)
            
            if args.gpu is not None:
                data = data.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(data)
            loss = criterion(output, target)
            train_loss += loss.item()
            
            # compute gradient and update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update pbar
            pbar.set_postfix(**{'loss (batch)': loss.item()})
            pbar.update()

    # print and write loss
    train_loss /= n_train
    print('Training loss: {}'.format(train_loss))
    writer.add_scalar('Loss/train', train_loss, epoch, args.gpu)

def validate(val_loader, model, criterion, writer, epoch, args):

    # switch to evaluate mode
    model.eval()

    val_loss = 0.0
    n_val = 0
    with tqdm(total=len(val_loader), desc="Validation", unit=' batch') as pbar:
        for batch in val_loader:
                   
            # data load
            data = batch["data"]
            target = batch["target"]
            n_val += len(data)
            
            if args.gpu is not None:
                data = data.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output   
            with torch.no_grad():
                output = model(data)
            val_loss += criterion(output, target).item()

            pbar.update()
            
    # print and write loss
    val_loss /= n_val
    print('Validation loss: {}'.format(val_loss))
    writer.add_scalar('Loss/val', val_loss, epoch, args.gpu)

    return val_loss

if __name__ == '__main__':
    main()
    
    """
    dimension = 2
    if (dimension == 1):
        train_path = "/hdd/hdd2/yong/L_FTS/1d_periodic/data1d_64_wp_diff/train"
        test_path = "/hdd/hdd2/yong/L_FTS/1d_periodic/data1d_64_wp_diff/eval"
        sample_file_name = "/hdd/hdd2/yong/L_FTS/1d_periodic/data1d_64_wp_diff/eval/fields_1_300000_000.npz"
    elif (dimension == 2):
        train_path = "/hdd/hdd2/yong/L_FTS/2d_periodic/data2d_64_wp_diff/train"
        test_path = "/hdd/hdd2/yong/L_FTS/2d_periodic/data2d_64_wp_diff/eval"
        sample_file_name = "/hdd/hdd2/yong/L_FTS/2d_periodic/data2d_64_wp_diff/eval/fields_1_300000_000.npz" 
      
    model = DeepFts(dimension, load_net="checkpoints_2/CP_epoch25.pth")
    
    sample_data = np.load(sample_file_name)
    nx = sample_data["nx"]

    X0 = sample_data["w_minus"]
    X1 = sample_data["g_plus"]
    Y  = sample_data["w_plus_diff"]
    Y_gen = model.generate_w_plus(X0, X1, nx)

    fig, axes = plt.subplots(2,2, figsize=(20,20))
    
    axes[0,0].plot(X0[:nx[0]])
    axes[1,0].plot(Y [:nx[0]])
    axes[1,0].plot(Y_gen[:nx[0]])
    #axes[1,1].plot(Y[0,0,:]-Y_gen[0,0,:])
     
    plt.subplots_adjust(left=0.2,bottom=0.2,
                        top=0.8,right=0.8,
                        wspace=0.2, hspace=0.2)
    plt.savefig('w_plus_minus.png')
    """
