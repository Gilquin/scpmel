# global imports
import os
import torch
# internal import
from .. import utils

__all__ = ["get_optimizer", "compute_gradients", "train", "validate"]


def get_optimizer(parameters, opt_args, resume=False, silent=True):
    """
    Method to instantiate the optimizer module either from scratch or
    from a checkpoint.

    Parameters
    ----------
    parameters : nn.Modules.parameters
        A generator of the model parameters.
        Example: model.parameters()
    opt_args : dict
        A dict containing the name of a torch.optim class and its
        arguments, for more infos see:
            https://pytorch.org/docs/stable/optim.html#torch.optim.Optimizer
        Example: {"name" : "Adam", "lr" : 0.001, "eps" : 1e-08}.
    resume : (boolean, optional)
        Whether to resume training from a checkpoint or not.

    Returns
    -------
    An instance of a torch.optim.Optimizer.

    """

    # get dict of trainable parameters
    train_params = filter(lambda p: p.requires_grad, parameters)

    # instantiate optimizer
    optimizer = getattr(torch.optim, opt_args["name"])
    if not resume:
        opt_args = utils.get_args(optimizer, opt_args)
        optimizer = optimizer(train_params, **opt_args)
    else:
        opt_args.pop("name")
        temp_args = utils.get_args(optimizer, opt_args["param_groups"][0])
        optimizer = optimizer(train_params, **temp_args)
        optimizer.load_state_dict(opt_args)

    if not silent:
        print("\nOptimizer parameters:")
        for (key, val) in utils.get_args(optimizer).items():
            print ("\t {}: {}".format(key, val))

    return optimizer


def compute_gradients(model, loader, loss_func, epoch, args, verbose, **kwargs):
    """
    Compute gradients of a torch.nn.Module over a whole dataset.

    Parameters
    ----------

    model : torch.nn.Modules
        The model containing the parameters from which to compute the gradients.
    loader : torch.utils.data.DataLoader
        The loader to iterate over the whole dataset batch-wise.
    loss_func : dict of torch.nn.modules.loss
        A dictionnary of loss functions to apply. The quantity of interest is
        obtained by sum reduction with equal weights for each loss function.
    epoch : int
        The epoch number.
    args : argparse.ArgumentParser
        The command line options.
    verbose : bool
        Wheter to print informations or not.

    Returns
    -------
    The vector of parameters gradients.
    """

    # zero out the model gradients
    model.zero_grad()

    # looping over epochs
    for batch_idx, batch in enumerate(loader):
        # extract data and targets
        data, targets, idx = batch
        # print("Batch {} in GPU {} have positions: {}".format(batch_idx, args.gpu, idx))
        # sent data and targets to device
        if torch.cuda.is_available() and (args.gpu is not None):
            data = data.cuda(args.gpu, non_blocking=True)
            targets = targets.cuda(args.gpu, non_blocking=True)
        # forward propagation
        outputs = model(data, **kwargs)
        # compute losses
        loss = sum([lfunc(outputs, targets) for lfunc in loss_func.values()])
        loss /= len(loss_func)
        # backward propagation to compute gradients
        loss.backward()

    return None # empty return


def train(model, loader, loss_func, optimizer, scheduler, epoch, args,
          verbose, grad_save, **kwargs):
    """
    Method to train the network. It performs batched stochastic gradient
    descent based on a user-defined optimizer.

    Parameters
    ----------

    model : torch.nn.Modules
        The model to train.
    loader : torch.utils.data.DataLoader
        The loader to extract the training dataset.
    loss_func : dict of torch.nn.modules.loss
        A dictionnary of loss functions to apply. The quantity of interest is
        obtained by mean reduction (equal weights for each loss function). Keys
        are used for the progressmeter statistics.
    optimizer : torch.optim.Optimizer
        The optimizer to perform the stochastic gradient descent (SGD).
    scheduler : torch.optim.lr_scheduler
        The scheduler to adjust the learning rate. Use None if no scheduler is
        required.
    epoch : int
        The epoch number.
    args : argparse.ArgumentParser
        The command line options.
    verbose : bool
        Wheter to print informations or not.
    grad_save : bool
        Wheter to save the per-batch gradients or not. 
    kwargs : (dict, optional)
        Additional parameters to pass to the model.

    Returns
    -------
    The loss average.
    """

    # set statistics to track
    losses = utils.AverageMeter("Loss", ':.3e')
    # set progress bar for main process
    if verbose:
        progress = utils.ProgressMeter(
            len(loader),
            args.print_freq,
            [losses],
            prefix="Epoch: [{}]".format(epoch)
            )
    else:
        progress = None

    # set the model in training mode
    model.train()

    # train over epochs
    for batch_idx, batch in enumerate(loader):
        # zero out the parameters gradient
        optimizer.zero_grad()
        # extract data and targets
        data, targets, _ = batch
        # sent data and targets to device
        if torch.cuda.is_available() and (args.gpu is not None):
            data = data.cuda(args.gpu, non_blocking=True)
            targets = targets.cuda(args.gpu, non_blocking=True)
        # forward propagation
        outputs = model(data, **kwargs)
        # # handle NaN
        # outputs[torch.isnan(outputs)] = 1
        # compute losses
        loss = sum([lfunc(outputs, targets) for lfunc in loss_func.values()])
        loss /= len(loss_func)
        # backward propagation to compute gradients
        loss.backward()
        # clip gradient to avoid explosion
        # torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=10)
        # adjust parameters
        optimizer.step()
        # adjust learning rate
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(loss.detach().item())
        elif scheduler is not None:
            scheduler.step()
        # record loss
        losses.update(loss.detach().item(), data.size(0))
        # print progression
        if (progress is not None) and (batch_idx % args.print_freq) == 0:
            progress.display(batch_idx)
        # save gradients
        if grad_save and verbose:
            gradients = torch.cat([param.grad.to('cpu').flatten() for param in model.parameters() if param.requires_grad])
            if args.distributed:
                gradients *= args.world_size
            torch.save(gradients, os.path.join(args.save_dir, "gradients_r{}e{}b{}.pt".format(args.run, epoch, batch_idx)))
    # stop progress
    if progress is not None:
        progress.finish()

    return losses.avg


def validate(model, loader, loss_func, args, verbose, **kwargs):
    """
    Method to evaluate the trained network on a dataset.

    Parameters
    ----------

    model : torch.nn.Modules
        The model to evaluate.
    loader : torch.utils.data.DataLoader
        The loader to extract the dataset.
    loss_func : dict of torch.nn.modules.loss
        A dictionnary of loss functions to apply. The quantity of interest is
        obtained by mean reduction (equal weights for each loss function). Keys
        are used for the progressmeter statistics.
    args : argparse.ArgumentParser
        The command line options.
    verbose : bool
        Wheter to print informations or not.
    kwargs : (dict, optional)
        Additional parameters to pass to the model.

    Returns
    ---------
    The loss mean.
    """

    # set statistics
    losses = utils.AverageMeter('Loss', ':.3e')
    # set progress bar for main process
    if verbose:
        progress = utils.ProgressMeter(
            len(loader),
            args.print_freq,
            [losses],
            prefix='Test: ')
    else:
        progress = None

    # set the model in training mode
    model.eval()

    # evaluate over epochs
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            # extract data and targets
            data, targets, _ = batch
            # sent data and targets to device
            if torch.cuda.is_available() and (args.gpu is not None):
                data = data.cuda(args.gpu, non_blocking=True)
                targets = targets.cuda(args.gpu, non_blocking=True)
            # compute predictions
            outputs = model(data, **kwargs)
            # compute losses
            loss = sum([lfunc(outputs, targets) for lfunc in loss_func.values()])
            loss /= len(loss_func)
            # record loss
            losses.update(loss.item(), data.size(0))
            # print progression
            if (progress is not None) and (batch_idx % args.print_freq) == 0:
                progress.display(batch_idx)

    # stop progress
    if progress is not None:
        progress.finish()

    return losses.avg