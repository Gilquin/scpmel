#!/usr/bin/env python3
"""
Created on Tue Mar 16 14:20:13 2021

@author: Laurent Gilquin

Template for distributed training of a Pytorch model.
This implementation is based on
[the official implementation](https://github.com/pytorch/examples/blob/master/imagenet/main.py)

Parts that need to be modified regarding the application considered are:
    * parser default options,
    * the loader file used for loading the model,
    * definitions of the loss function, the optimizer and the scheduler,
    * loading the datasets.
"""

# system modules
import os
# utility modules
import argparse, gc
# torch submodules
import torch
from torch.utils.data.distributed import DistributedSampler
# from torch.optim import lr_scheduler
# torch parallel submodules
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
# scpmel modules
from scpmel.sampling import fix_all_seeds
from scpmel.training import (H5Dataset, KineticLoss, get_optimizer, resume_from_checkpoint,
                             save_checkpoint, train, validate)
from scpmel.utils import Logger, cpuStats, mkdir_p, save_gradflow
# relative import
from loader import load_model

# set torch default dtype to double precision
torch.set_default_dtype(torch.float64)


# parser setting with default values and actions
parser = argparse.ArgumentParser(description='PyTorch: model training')
parser.add_argument('file', metavar='DIR',
                    help="""The path to the hdf5 file used for both instantiating
                    the model and loading the dataset.""")
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help="""The number of workers use in the DataLoader.""")
parser.add_argument('--run', type=str, default='', metavar='RUN',
                    help="""The run name, useful for saving (default none).""")
parser.add_argument('--epochs', default=40, type=int, metavar='N',
                    help="""The number of total epochs to run.""")
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help="""Starting epoch number, useful for restarts.""")
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help="""This is the total batch size of all GPUs 
                    on the current node when using Distributed Data Parallel. 
                    The correspoding batch size fed to each GPU of the node is 
                    divided by the world-size.""")
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help="""The print frequency of batch results per
                    epoch (default: 10).""")
parser.add_argument('-c', '--checkpoint_dir', default='checkpoints', type=str, metavar='PATH',
                    help="""The path to the checkpoints directory (default: checkpoints).""")
parser.add_argument('--save_dir', default='gradients', type=str, metavar='PATH',
                    help="""The path where the gradients and samples will be saved
                    (default: gradients).""")
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help="""The path to the latest checkpoint (default: none).""")
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help="""Flag to only evaluate the model on the validation set.""")
parser.add_argument('--world-size', default=-1, type=int,
                    help=""""The number of nodes for the distributed training.""")
parser.add_argument('--rank', default=-1, type=int,
                    help=""""The node rank for the distributed training.""")
parser.add_argument('--dist-url', default='', type=str,
                    help="""The url used to set up the distributed training.""")
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help=""" The distributed communication backend.""")
parser.add_argument('--seed', default=None, type=int,
                    help="""The seed for reproducibility.""")
parser.add_argument('--gpu', default=None, type=int,
                    help="""The GPU id to use. If None, default to distribution over
                    all available GPU of the node.""")
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help="""Use multi-processing distributed to launch N processes per node,
                    which has N GPUs.""")


# record best loss
best_loss = 1e6

# main function
def main():

    # get the command line options
    args = parser.parse_args()

    # build checkpoints directory
    if not os.path.isdir(args.checkpoint_dir):
        print("Creating the directory where the checkpoints will be saved.")
        mkdir_p(args.checkpoint_dir)
    # build figures subdirectory
    print("Creating the directory where the figures will be saved.")
    if not os.path.isdir(args.checkpoint_dir+"/figures"):
        mkdir_p(args.checkpoint_dir+"/figures")
    # build gradients directory
    if not os.path.isdir(args.save_dir):
        print("Creating the directory where the gradients will be saved.")
        mkdir_p(args.save_dir)

    # get the number of nodes for distributed training
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    # set distributed flag
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    # get number of available GPUs
    ngpus_per_node = torch.cuda.device_count()
    print("The number of available GPU is {}".format(ngpus_per_node))

    # define workers
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)

    # manually clean cyclic references
    gc.collect()


# per-process main function
def main_worker(gpu, ngpus_per_node, args):

    global best_loss
    args.gpu = gpu

    # fix seed for reproducibility
    if args.seed is not None:
        _ = fix_all_seeds(args.seed)

    if args.gpu is not None:
        print("Using GPU: {} for model training.".format(args.gpu))

    # set rank
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # print condition
    io_check = not args.multiprocessing_distributed or (args.multiprocessing_distributed 
    and args.rank % ngpus_per_node == 0)

    # load model
    if io_check:
        print("=> Loading model from:\t'{}'".format(args.file))
    model = load_model(args.file, "cutoff1", 25.0, io_check)

    # send the model to the device
    if io_check:
        print("Sending the model to the device(s).")
    if not torch.cuda.is_available():
        print('Cuda is not available, falling back to CPU (computation will be very slow).')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            model.set_device(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        model.set_device(args.gpu)
    else:
        pass

    # define loss function (criterion)
    if io_check:
        print("Defining the loss functions.")
    loss_func = {
        'L2Loss': torch.nn.MSELoss(reduction="mean"),
        'KLoss': KineticLoss(reduction="mean")
        }

    # define optimizer
    if io_check:
        print("Defining the optimizer.")
    opt_args = {
        "name": "RAdam",
        "lr": 1e-3, # 0.001 is the default value for RAdam
        }
    optimizer = get_optimizer(model.parameters(), opt_args)

    # load dataset
    if io_check:
        print("Loading the dataset.")
    dset = H5Dataset(args.file, name="cutoff1")
    dset.set_mode("training")
    # define sampler (distributed or not)
    if args.distributed:
        train_sampler = DistributedSampler(dset, shuffle=True, drop_last=True)
    else:
        train_sampler = None
    #instantiate DataLoader for training
    if io_check:
        print("Instantiating DataLoaders for training and validation.")
    train_loader = torch.utils.data.DataLoader(
        dset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler,
        drop_last=True)
    #instantiate DataLoader for validation
    val_loader = torch.utils.data.DataLoader(
        dset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # set lr scheduler
    if io_check:
        print("Instantiating learning rate scheduler.")
    scheduler = None

    # optionally resume from a checkpoint
    if args.resume:
        best_loss = resume_from_checkpoint(model, optimizer, args)
        if args.distributed:
            dist.barrier() # barrier to ensure that all process have loaded the checkpoint
        else:
            if io_check:
                print("=> No checkpoint found at '{}'".format(args.resume))

    # enable the inbuilt cudnn auto-tuner to find the best algorithm to use for the hardware
    cudnn.benchmark = True

    # evaluate mode
    if args.evaluate:
        if io_check:
            print("Starting model evaluation.")
        dset.set_mode("validation")
        valid_loss = validate(model, val_loader, loss_func, args, io_check)
        print("validation loss: {}".format(valid_loss))
        return None

    # training mode
    if io_check:
        print("Starting model training.")

    if io_check:
        # define logger
        print("Defining logger.")
        title = 'run_' + args.run
        path_log = os.path.join(args.checkpoint_dir, title + 'log.txt')
        logg = Logger(path_log, title)
        logg.set_names(['Learning Rate', 'Train Loss', 'Valid Loss'])

    for epoch in range(args.start_epoch, args.epochs):
        # set epoch
        if args.distributed:
            # required to make shuffling work properly
            train_sampler.set_epoch(epoch)
        # train for one epoch
        dset.set_mode("training")
        train_loss = train(model, train_loader, loss_func, optimizer, scheduler,
                           epoch, args, io_check, grad_save=True)
        # evaluate on validation set
        dset.set_mode("validation")
        valid_loss = validate(model, val_loader, loss_func, args, io_check)
        # remember best loss average
        is_best = valid_loss < best_loss
        best_loss = min(valid_loss, best_loss)

        # I/0 operations on main process
        if io_check:
            # save checkpoint
            print("Saving checkpoint, storing gradflow and logger.")
            save_path = os.path.join(args.checkpoint_dir, "checkpoint{}.pth.tar".format(args.run))
            save_checkpoint({
                'epoch': epoch + 1,
                'file': args.file,
                'state_dict': model.state_dict(),
                'best_loss': best_loss,
                'optimizer': optimizer.state_dict(),
            }, is_best, filename=save_path)
            # update logger
            logg.append([optimizer.param_groups[0]['lr'], train_loss, valid_loss])
            # save gradflow
            save_gradflow(model, args.checkpoint_dir+"/figures/gradflow_{}".format(epoch))
            # log memory usage
            cpuStats(args.checkpoint_dir+"/mem_log.txt", epoch==args.start_epoch)
            if args.distributed:
                dist.barrier() # wait for the main process to finish operations

    # close logger
    if io_check:
        logg.close()

    # cleanup
    if args.distributed:
        dist.destroy_process_group()

    return None


if __name__ == '__main__':
    import time
    from datetime import timedelta
    start = time.time()
    main()
    elapsed = timedelta(seconds=time.time() - start)
    print("Total elapsed time is:\t{}".format(str(elapsed)))
