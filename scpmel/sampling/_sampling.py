# global imports
from scipy.ndimage import convolve
import numpy as np
import random
import torch
import torch.backends.cudnn as cudnn
import warnings

__all__ = ["aldama_samples", "barsinai_samples", "custom_samples",
           "fix_all_seeds", "mvn_samples", "svd_samples"]


def fix_all_seeds(seed, cuda_flag=False):
    """
    Utility function to fix the seeds in all the modules (numpy, random, etc ...)
    """
    random.seed(seed)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available() and cuda_flag:
        cudnn.deterministic = True
        warnings.warn(
            'You have chosen to seed training.\n'
            'This will turn on the CUDNN deterministic setting, '
            'which can slow down the training considerably!\n'
            'You may see unexpected behavior when restarting from checkpoints.')

    return rng


def convolution_samples(xx, kernel, seed=None, **kwargs):
    """
    Draw samples from kernel using discrete convolution.

    Parameters
    ----------
    xx : array like
        The vector of spatial points.
    kernel : callable
        The kernel function used as the convolution filter.
    dx : float
        The sampling frequency.
    seed: bool (default False)
        Whether to fix the seed for reproducibility.

    Returns
    -------
    The vector of samples.
    """

    # set seed
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng(None)

    # center samples
    xx = xx-(xx[-1]-xx[0])/2
    dx = xx[1] - xx[0]
    weights = kernel(xx, **kwargs) * np.sqrt(dx)
    res = convolve(
        rng.standard_normal((len(xx)), weights))

    return res


def mvn_samples(xx, covfunc, seed=None, **kwargs):
    """
    Draw samples from np.multivariate.normal function given covariance function.

    Parameters
    ----------
    xx : array like
        The vector of spatial points.
    covfunc : callable
        The covariance function used to defined the covariance matrix.
    seed: bool (default False)
        Whether to fix the seed for reproducibility.

    Returns
    -------
    The vector of samples.
    """

    # set seed
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng(None)

    # covariance matrix
    Cx = covfunc(np.expand_dims(xx, 1) - xx, **kwargs)
    # unconditional samples
    res = rng.multivariate_normal(
        mean=np.zeros(len(xx)), cov=Cx, size=1)[0]

    return res


def svd_samples(xx, covfunc, reduce=False, seed=None, **kwargs):
    """
    Draw samples using SVD decomposition of the covariance matrix
    with truncature based on the Frobenius norm of the eigenvalues.

    Parameters
    ----------
    xx : array like
        The vector of spatial points.
    covfunc : callable
        The covariance function used to defined the covariance matrix.
    seed: bool (default False)
        Whether to fix the seed for reproducibility.

    Returns
    -------
    The vector of samples.
    """

    # set seed
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng(None)

    # covariance matrix
    Cx = covfunc(np.expand_dims(xx, 1) - xx, **kwargs)
    U, vp, _ = np.linalg.svd(Cx, hermitian=True)
    if reduce:
        # assess the number of signicant components
        nbcomps = np.argmax(np.cumsum(vp)/np.sum(vp)>(1-1e-12))
        vp = vp[:nbcomps]
        U = U[:, :nbcomps]
    else:
        nbcomps = len(xx)
    # unconditional samples
    res = (np.sqrt(vp) * U) @ rng.standard_normal(nbcomps)

    return res


def aldama_samples(xx, spectrum, sigma, Nk, cutoff, seed=None, **kwargs):
    """
    The sampling procedure as defined in
    [Aldama, Alvaro 1990](https://www.springer.com/gp/book/9783540521372)
    Section 5.7 (p.149).

    Parameters
    ----------
    xx : array like
        The spatial position.
    spectrum : callable
        The energy spectrum function.
    sigma : float
        the energy spectrum variance.
    Nk : int
        the number of wavenumber bands.
    cutoff : float
        The cutoff value for the wavenumber.
    seed: bool (default False)
        Whether to fix the seed for reproducibility.

    Returns
    -------
    The vector of samples.
    """

    # set seed
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng(None)

    delta_k = cutoff / Nk
    delta_kp = delta_k / 20
    kk = (np.arange(1, Nk+1) - 1/2) * delta_k
    kkp = kk + rng.random((Nk),) * delta_kp - delta_kp / 2
    phi = rng.random((Nk),) * 2*np.pi

    vv = (spectrum(kk, sigma=sigma, **kwargs)*delta_k)**(1/2)
    res = np.cos(xx[:,np.newaxis] @ kkp[np.newaxis,:] + phi) @ vv

    return 2 * res


def barsinai_samples(xx, N=20, lset=[3,4,5,6], seed=None):
    """
    The sampling procedure as defined in the Supplementary Material of
    [BarSinai 2019](https://doi.org/10.1073/pnas.1814058116)
    that generates sum of long-wavelength sinusoidal functions.

    Parameters
    ----------
    xx : array like
        The spatial position.
    N : int
        the number of summands.
    lset : list
        A set of phase frequencies from which to draw.
    seed: bool (default False)
        Whether to fix the seed for reproducibility.

    Returns
    -------
    The vector of samples.
    """

    # set seed
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng(None)

    xrange = xx[-1] - xx[0]
    A = rng.random((N,)) - 0.5
    phi = rng.random((N,)) * 2*np.pi
    ell = rng.choice(lset, size=N, replace=True)

    return (A[:,np.newaxis] * np.sin(2*np.pi/xrange * ell[:,np.newaxis] * xx[np.newaxis,:] + phi[:,np.newaxis])).sum(axis=0)


def custom_samples(xx, amp=100.0, N=20, lset=[3,4,5,6], seed=None):
    """
    Custom sampling procedure that generates a sum of long-wavelength sinusoidal functions
    negative, bounded and converging to 0 at the edges of the domain. The convergence to
    0 is enforced through and exponential term.

    Parameters
    ----------
    xx : array like
        The spatial position.
    amp : float
        The negative bound for the signal amplitude.
    N : int
        the number of summands.
    lset : list
        A set of phase frequencies from which to draw.
    seed: bool (default False)
        Whether to fix the seed for reproducibility.

    Returns
    -------
    The vector of samples.
    """

    # set seed
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng(None)

    xrange = xx[-1] - xx[0]
    low, upp = np.quantile(xx, [0.4, 0.6])
    A = rng.random((N,)) - 0.5
    phi = rng.random((N,)) * 2*np.pi
    ell = rng.choice(lset, size=N, replace=True)

    # reset seed
    if seed is not None:
        np.random.seed(None)

    res = (A[:,np.newaxis] * np.sin(2*np.pi/xrange * ell[:,np.newaxis] * xx[np.newaxis,:] + phi[:,np.newaxis]) - 1).sum(axis=0)
    res *= amp/N*np.exp(-(xx/(upp-low))**2)

    return res