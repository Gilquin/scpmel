# global imports
import os
import shutil
import torch
# internal import
from .. import utils

__all__ = ["resume_from_checkpoint", "save_checkpoint"]


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        path = os.path.dirname(filename)
        shutil.copyfile(filename, os.path.join(path,'model_best.pth.tar'))


def resume_from_checkpoint(model, optimizer, args, verbose):
    """
    Instantiate the model and the optimizer from a checkpoint.

    Parameters
    ----------

    model : torch.nn.Modules
        The model to train.
    loader : torch.utils.data.DataLoader
        The loader to extract the training dataset.
    optimizer : torch.optim.Optimizer
        The optimizer to perform the stochastic gradient descent (SGD).
    verbose : bool
        Wheter to print informations or not.

    Returns
    -------
    None
    """

    if os.path.isfile(args.resume):
        # load checkpoint data
        if verbose:
            print("=> Resuming from checkpoint '{}'".format(args.resume))
        if args.gpu is None:
            checkpoint = torch.load(args.resume)
        else:
            # Map model to be loaded to specified single gpu
            loc = 'cuda:{}'.format(args.gpu)
            checkpoint = torch.load(args.resume, map_location=loc)
        args.start_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        # cast best loss to tensor
        if not torch.is_tensor(best_loss):
            best_loss = torch.tensor(
                best_loss, dtype=torch.get_default_dtype())
        if args.gpu is not None:
            # best_loss may be from a checkpoint from a different GPU
            best_loss = best_loss.to(args.gpu)
        # format state dict for parallel compatibility
        model.load_state_dict(
            utils.format_state_dict(checkpoint['state_dict'], args.multiprocessing_distributed))
        optimizer.load_state_dict(checkpoint['optimizer'])
        if args.lr < optimizer.param_groups[0]['lr']:
            optimizer.param_groups[0]['lr'] = args.lr
        if verbose:
            print("=> Loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

    return best_loss