#!/usr/bin/env python3
"""
@author: Laurent Gilquin

Test script for gradient accumulation: local GPU vs distributed single node multi GPU using the
DistributedDataParallel (DDP) module of Pytorch.

Particularities of the present script:
    * 'sum' reduction is used for the loss so thzt gradients are averaged once all the dataset batches have been processed,
    * in the distributed case, since the local Reducer created by the DDP module first use an "allreduce" operation to sum the
    gradients and then divide the result by the world_size (e.g. the number of GPUS), the resulting .grad attribute must be
    multiply by the world_size at the end of the epoch to recover the correct gradient summed over the whole dataset.

For more informations see the following documentation/discussion:
        https://pytorch.org/docs/stable/notes/ddp.html (section backward pass)
        https://discuss.pytorch.org/t/gradient-aggregation-in-distributeddataparallel/101715/3

Example is inspired from:
    https://github.com/pytorch/examples/blob/master/distributed/ddp/example.py

"""

# I/O modules
import argparse
import os
# torch modules
import torch
import torch.nn as nn
# torch distributed modules
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.elastic.multiprocessing.errors import record
# torch transverse modules
from torch.utils.data import Dataset
from torch.nn.modules.loss import _Loss


# parser
parser = argparse.ArgumentParser("Pytorch gradient consistency through DDP module.")
parser.add_argument("--batch_size", type=int, default=100)
parser.add_argument("--ndata", type=int, default=100)
parser.add_argument('--use-ddp', default=False, action='store_true')

# some utility classes

# basic dataset compatible with DistributedSampler
class BasicDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return (self.data[idx,0,:], self.data[idx,1,:], idx)

# the neural network toy model
class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 10)
        self.shape = (10,)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))

# a custom loss function
class KineticLoss(_Loss):

    def __init__(self, reduction='mean'):
        super(KineticLoss, self).__init__(None, None, reduction)
        if self.reduction not in ["none", "sum", "mean"]:
            raise ValueError("{} is not a valid value for reduction".format(reduction))

    def forward(self, outputs, targets):
        res = torch.abs((outputs**2).mean(tuple(range(2, targets.ndim))) -\
                        (targets**2).mean(tuple(range(2, targets.ndim)))).sum(axis=-1)
        if self.reduction == "mean":
            return torch.mean(res)
        elif self.reduction == "sum":
            return torch.sum(res)
        else:
            return res



def compute_gradients(model, loader, loss_func, devices):
    """Function that iterates over batches and accumulates gradient."""
    model.zero_grad()
    # loop over batch
    for batch_idx, batch in enumerate(loader):
        # extract data and targets
        data, targets, idx = batch
        # sent data and targets to device
        data = data.cuda(devices[0], non_blocking=True)
        targets = targets.cuda(devices[0], non_blocking=True)
        # forward propagation to build the autograd graph
        outputs = model(data)
        # compute losses
        loss = sum([lfunc(outputs, targets) for lfunc in loss_func.values()])
        loss /= len(loss_func)
        print("Device:\t{}, batch:\t{}, indices:\t{}, loss:\t{}".format(
            devices[0], batch_idx, idx, loss.detach().item()))
        # backward propagation to accumulate gradients
        loss.backward()


def demo_basic(local_rank, local_world_size, args):

    
    n = torch.cuda.device_count() // local_world_size
    device_ids = list(range(local_rank * n, (local_rank + 1) * n))
    print(
        f"[{os.getpid()}] rank = {dist.get_rank()}, "
        + f"world_size = {dist.get_world_size()}, n={n}, device_ids = {device_ids}"
    )

    # only printing infos on main process
    verbose = device_ids[0] == 0

    # send ddp model to device
    model = ToyModel().cuda(device_ids[0])
    if args.use_ddp:
        if verbose:
            print("Using DistributedDataParallel module.")
        model = DDP(model, device_ids)

    # define loss functions with sum reduction
    loss_func = {
        'L2Loss': torch.nn.MSELoss(reduction="sum"),
        'KLoss': KineticLoss(reduction="sum")
        }

    # random dataset for illustration purpose
    dset = BasicDataset(torch.randn((args.ndata,2,10)))
    if verbose:
        print("dataset length equals:\t{}".format(len(dset)))

    # reduce batch size according to number of devices
    batch_size = int(args.batch_size / local_world_size)
    if verbose:
        print("batch_size equals:\t{}".format(batch_size))
        
    # instantiate DataLoader
    if local_world_size>1:
        sampler = DistributedSampler(dset, shuffle=False, drop_last=True)
    else:
        sampler = None
    if verbose:
        print("Instantiating DataLoader for batch iterations.")
    dataloader = torch.utils.data.DataLoader(
        dset, batch_size=args.batch_size, shuffle=False, drop_last=True,
        num_workers=4, pin_memory=True, sampler=sampler)
    if local_world_size>1:
        len_dset = dataloader.batch_size * (sampler.total_size // dataloader.batch_size)
    else:
        len_dset = dataloader.batch_size * (len(dataloader.dataset) // dataloader.batch_size)
    if verbose:
        print("Dataset length equals:\t{}".format(len_dset))
    
    # compute the gradient
    compute_gradients(model, dataloader, loss_func, device_ids)

    # recover the gradients
    gradients = torch.cat([param.grad.to('cpu').flatten() for param in model.parameters() if param.requires_grad])

    # handle the ddp Reducer default allreduce by multiplying by the world_size
    if local_world_size>1:
        gradients *= local_world_size
    if args.use_ddp:
        gradients /= (len_dset * model.module.shape[0])
    else:
        gradients /= (len_dset * model.shape[0])
    print("Accumulated gradients equal:\t {}".format(gradients))

    return None

@record
def main():
    # check if cuda is initialized
    if not torch.cuda.is_initialized():
        torch.cuda.init()
    # get parser arguments
    args = parser.parse_args()
    # get local rank and world size from environnement variables
    local_rank = int(os.environ['LOCAL_RANK'])
    local_world_size = int(os.environ['LOCAL_WORLD_SIZE'])
    # These are the parameters used to initialize the process group
    env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
    }
    print(f"[{os.getpid()}] Initializing process group with: {env_dict}")
    dist.init_process_group(backend="nccl")
    print(
        f"[{os.getpid()}]: world_size = {dist.get_world_size()}, "
        + f"rank = {dist.get_rank()}, backend={dist.get_backend()}"
    )

    # Explicitly setting seed to make sure that models created in two processes
    # start from same random weights and biases.
    torch.manual_seed(404)

    demo_basic(local_rank, local_world_size, args)

    # Tear down the process group
    dist.destroy_process_group()

if __name__ == "__main__":  
    main()
