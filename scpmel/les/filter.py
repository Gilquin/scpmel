# global imports
import torch
import numpy as np
# relative import
from .. import training
from .misc import _to_tensor


__all__ = ["get_filter", "filtr"]


def get_filter(xx, width, choice="Gaussian", gamma=6.0, tol=1e-12, device='cpu'):
    """
    The ideal (low pass), box and gaussian filters as implemented in
    [Aldama, Alvaro 1990](https://www.springer.com/gp/book/9783540521372)

    Parameters
    ----------
    xx : array like
        The vector of spatial points
    width : float
        The filter characteristic width
    choice : string
        The type of filter, one of {'LowPass', 'Box', 'Gaussian'}
    gamma : float
        Constant parameter only relevant when choice='Gaussian'. Default value
        equals 6.0 for historical reasons (same Leonard approximation
        as the box filter with this value).
    tol : float
        The tolerance for the kernel support (Default to 1e-12)

    Returns
    -------
    The filter values.
    """

    # handle choice
    if choice not in ["LowPass", "Box", "Gaussian"]:
        raise ValueError("choice must be one of {'LowPass', 'Box', 'Gaussian'}.")
    if choice=="LowPass":
        cutoff = np.pi/width
        out = np.sin(cutoff*xx)/(cutoff*(xx+1e-40))
    elif choice=="Box":
        out = 1/width * (np.abs(xx) < width/2)
    else:
        out = np.sqrt(gamma/np.pi) * np.exp(-gamma * (xx/width)**2)/width
    # contract the filter to keep only relevant weights
    idx = np.where(np.abs(out) > tol)[0]
    xx = xx[idx]
    out = out[idx]

    return xx, out/out.sum()


def filtr(x, weights, device='cpu'):
    """
    Convolve a signal with periodic boundaries.

    Parameters
    ----------
    x : array like
        The signal to filter
    weights : array like
        The filter weights (normalized)

    Returns
    -------
    The filtered signal as a torch tensor
    """

    # move array to tensor
    x = _to_tensor(x, device)

    # check if number of weights is odd
    if len(weights) % 2 == 0:
        weights = np.r_[weights, 0.0] # add a zero to the right
    kernel = torch.tensor(
        weights, dtype=torch.get_default_dtype()).reshape((1,1)+(len(weights),))
    torch_conv = training.ConvLayer(kernel)

    # reshape x if necessary
    if len(x.shape) != len(kernel.shape):
        x = torch.reshape(x, (1,1)+(len(x),))

    return torch_conv(x)
