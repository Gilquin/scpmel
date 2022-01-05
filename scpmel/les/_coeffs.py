# global import
import findiff
import torch

def get_stencil(dx, order, acc, scheme="center"):
    """
    Utility function to retrieve the stencil coefficients of a discretized
    partial derivative.

    Parameters
    ----------
    dx : float
        The spatial step.
    order : int
        The partial derivative order.
    acc : int
        The partial derivative accuracy order.       
    scheme : string
        One of 'center', 'forward', 'backward'

    Returns
    -------
    The stencil coefficients as a torch.Tensor.
    """

    # get the coefficients using findiff package
    coeffs = findiff.coefficients(order, acc, offsets=None, symbolic=False)
    # build the stencil
    stencil = torch.from_numpy(coeffs[scheme]["coefficients"] / dx)

    return stencil.reshape((1,1,-1))