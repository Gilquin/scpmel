#!/usr/bin/env python3
"""
@author: Laurent Gilquin

Template for multi node multi GPU distributed computation of a Pytorch model gradients.
In both case, for each epoch the procedure proceeds as follows:
    * a new vector of parameters is sampled (currently based on the default distribution),
    * a forward call of the model is performed to compute the loss with respect to the vector of parameters
    and construct the DAG (Directly Acyclic Grapgh),
    * a backward call is performed to collect the Jacobian matrix through the Pytorch backprogration algorithm.
    * the resulting gradients are stored (by the main process in the distibuted case) and save to the specified
    directory.

Particularities of the current script :
    * 'sum' reduction is used for the loss which ensure that gradients can be averaged once all the dataset batches have been processed,
    * in the distributed case, since the local Reducer created by the DistributedDataParallel module first use an "allreduce"
    to sum the gradients and then divide the result by the world_size (e.g. the number of GPUS), the resulting .grad attribute
    must be multiply by world_size at the end of the epoch to recover the gradient summed over the whole dataset.
"""

# system module
import os
# utility modules
import argparse, gc
# torch submodules
import torch
from torch.utils.data.distributed import DistributedSampler
# torch parallel submodules
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
# scpmel modules
from scpmel.utils import mkdir_p
from scpmel.training import H5Dataset, KineticLoss, compute_gradients, reset_params
from scpmel.sampling import fix_all_seeds
# relative import
from loader import load_model

# set torch default dtype to double precision
torch.set_default_dtype(torch.float64)

# parser setting with default values and actions
parser = argparse.ArgumentParser(description='PyTorch: dimensionality reduction')
parser.add_argument('file', metavar='DIR',
                    help="""The path to the hdf5 file used for both instantiating the
                    model and loading the dataset.""")
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help="""The number of workers use in the DataLoader.""")
parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                    help="""Number of parameter samples from which to evaluate gradients. This 
                    is equivalent to the number of epochs in a standard training procedure, as
                    a new vector of parameters is sampled at the start of each epoch.""")
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help="""Starting epoch number, useful for restarts.""")
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help="""This is the total batch size of all GPUs 
                    on the current node when using Distributed Data Parallel. 
                    The correspoding batch size fed to each GPU of the node is 
                    divided by the world-size.""")
parser.add_argument('--save_dir', default='gradients', type=str, metavar='PATH',
                    help="""The path where the gradients and samples will be saved.""")
parser.add_argument('--world-size', default=-1, type=int,
                    help=""""The number of nodes for the distributed inmplementation.""")
parser.add_argument('--rank', default=-1, type=int,
                    help="""The node rank for the distributed inmplementation.""")
parser.add_argument('--dist-url', default='env://', type=str,
                    help="""The url used to set up the distributed inmplementation.""")
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help=""" The distributed communication backend.""")
parser.add_argument('--seed', default=None, type=int,
                    help="""The seed for initializing the parameters samples.""")
parser.add_argument('--gpu', default=None, type=int,
                    help="""The GPU id to use. If None, default to distribution over
                    all available GPU of the node.""")
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help="""Use multi-processing distributed to launch M processes per node,
                    which has N GPUs.""")


# main function
def main():

    # check that cuda is available
    if not torch.cuda.is_available():
        raise("""The Pytorch module must be installed with CUDA compatibility, please consult the official documentation at:\n
                 \t https://pytorch.org/""")

    # get the command line options
    args = parser.parse_args()

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
    if ngpus_per_node == 0:
        raise("""There is no available GPU.""")
    else:
        print("The number of available GPU is {}".format(ngpus_per_node))

    # define workers
    if args.multiprocessing_distributed:
        # additional debug logging when models trained with
        os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
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

    # fix seed to guarantee that each distributed model will get the same samples
    if args.seed is not None:
        _ = fix_all_seeds(args.seed)

    args.gpu = gpu
    print("Using GPU: {} for gradient computation".format(args.gpu))

    # set rank
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed, rank needs to be the
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
    model = load_model(args.file, "cutoff1", 25.0, verbose=io_check)

    # send the model to the device
    if io_check:
        print("Sending the model to the device(s).")
    if args.distributed:
        print("Instantiating the DistributedDataParallel module.")
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.to(args.gpu)
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
        if io_check:
            print("Model shape is:\t{}".format(model.module.shape[0]))
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.to(args.gpu)
        if io_check:
            print("model shape is:\t{}".format(model.shape[0]))
    if io_check:
        print("world_size is:\t{}\nbatch_size is:\t{}".format(args.world_size, args.batch_size))

    # define loss function (criterion)
    if io_check:
        print("Defining the loss functions.")
    loss_func = {
        'L2Loss': torch.nn.MSELoss(reduction="sum"),
        'KLoss': KineticLoss(reduction="sum")
        }

    # load dataset
    if io_check:
        print("Loading the dataset.")
    dset = H5Dataset(args.file, name="cutoff1")
    dset.set_mode("training")
    # define sampler (distributed or not)
    if args.distributed:
        train_sampler = DistributedSampler(dset, shuffle=False, drop_last=True)
    else:
        train_sampler = None
    #instantiate DataLoader
    if io_check:
        print("Instantiating DataLoader for batch iterations.")
    dataloader = torch.utils.data.DataLoader(
        dset, batch_size=args.batch_size, shuffle=False, drop_last=True,
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    # the actual dataset length caused by drop_last=True
    if args.distributed:
        len_dset = dataloader.batch_size * (train_sampler.total_size // dataloader.batch_size)
    else:
        len_dset = dataloader.batch_size * (dataloader.dataset.dataset_len // dataloader.batch_size)
    if io_check:
        print("dataset length is:\t{}".format(len_dset))


    # enable the inbuilt cudnn auto-tuner to find the best algorithm to use for the hardware
    cudnn.benchmark = True

    # Gradient computations over samples
    for epoch in range(args.start_epoch, args.start_epoch+args.epochs):
        if io_check:
            print("Begin gradient computations for sample:\t{}".format(epoch))
        # sample a new vector of parameters
        model.apply(reset_params)
        # compute the gradients
        compute_gradients(model, dataloader, loss_func, epoch, args, io_check)
        # save the samples and gradients
        if io_check:
            # save the samples
            samples = torch.cat([param.detach().to('cpu').flatten() for param in model.parameters() if param.requires_grad])
            save_path = os.path.join(args.save_dir, "samples{}.pt".format(epoch))
            torch.save(samples, save_path)
            # save the gradients
            gradients = torch.cat([param.grad.to('cpu').flatten() for param in model.parameters() if param.requires_grad])
            # handle the ddp Reducer default allreduce and divive the gradients by the total number of data
            if args.distributed:
                gradients *= args.world_size
                gradients /= (len_dset * model.module.shape[0])
            else:
                gradients /= (len_dset * model.shape[0])
            save_path = os.path.join(args.save_dir, "gradients{}.pt".format(epoch))
            torch.save(gradients, save_path)
        # add a barrier to make sure the gradients recovered are unaffected
        if args.distributed:
            dist.barrier()

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
