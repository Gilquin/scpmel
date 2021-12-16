# global imports
import torch
import numpy as np
import scipy.signal as ssp
# relative import
from .misc import _to_tensor

__all__ = ["correlate", "energy_spectrum"]


def _next_fast_len(n, factors=[2, 3, 5, 7]):
    """
      Returns the minimum integer not smaller than n that can
      be written as a product (possibly with repettitions) of
      the given factors.
    """

    best = float('inf')
    stack = [1]
    while len(stack):
        a = stack.pop()
        if a >= n:
            if a < best:
                best = a
                if best == n:
                    break; # no reason to keep searching
        else:
            for p in factors:
                b = a * p
                if b < best:
                    stack.append(b)
    return best


def correlate(x, y, optimized = True, device='cpu'):
    """
    Cross-correlation of two signals through torch.fft

    Parameters
    ----------
    x : array like
        The first signal
    y : array like
        The second signal
    weights : array like
        The filter weights (normalized)

    Returns
    -------
    The filtered signal as a torch tensor
    """

    # move arrays to tensor
    x = _to_tensor(x)
    y = _to_tensor(y)

    # output target length of crosscorrelation
    length = len(x) + len(y) - 1
    if optimized:
        length = _next_fast_len(length, [2, 3])

    # the last signal_ndim axes (1,2 or 3) will be transformed
    fft_x = torch.fft.rfft(x, length, dim=-1)
    fft_y = torch.fft.rfft(y, length, dim=-1)

    # take the complex conjugate of one of the spectrums
    fft_cross = torch.conj(fft_x) * fft_y

    # back to time domain
    x_cross = torch.fft.irfft(fft_cross, n=length, dim=-1)

    # shift the signal to make it look like a proper crosscorrelation,
    # and transform the output to be purely real
    out = torch.roll(x_cross, length // 2)

    return out


def autocorr(x, maxlag=None, device='cpu'):
    """
    Unbiased estimator of the autocorrelation of a signal
    http://stackoverflow.com/q/14297012/190597
    http://en.wikipedia.org/wiki/Autocorrelation#Estimation
    """

    N = len(x)
    # correlate
    res = correlate(x, x, device=device)[-N:]
    # handle maxlag
    if maxlag is None:
        maxlag = N
    res = res[:int(maxlag)]
    res = res / torch.arange(N, N-maxlag, -1)

    return res


def energy_spectrum(signal, length, nperseg=None):
    """
    Computes the energy spectrum of a signal by dividing it into overlapping
    segments, computing a modified periodogram for each segment and
    averaging the periodograms.
    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html
    for the full documentation.
    
    Parameters
    ----------
    signal : array like
        The signal from which to estimate the energy spectrum.
    length : int
        The spatial domain length
    nperserg : int
        the number of segments.

    Returns
    -------
    The estimated energy spectrum.
    """

    # tensor to array
    if torch.is_tensor(signal):
        signal = signal.numpy()

    # length signal
    npts = len(signal)
    if nperseg is None:
        nperseg = npts
    # get normalization factor (depending)
    factor =  2 * int(npts // nperseg)
    # call welch function to estimate power spectrum
    freqs, Pt = ssp.welch(
        signal, fs=npts, window="boxcar", nperseg=nperseg, 
        noverlap=0, nfft=npts, detrend=None, return_onesided=True,
        scaling='spectrum', axis=-1, average='mean')
    # power spectrum
    out =  length/factor*Pt[:-1]

    return freqs, out


def __main__():

    import matplotlib.pyplot as plt
    plt.style.use('paper')
    plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
    from .filter import get_filter

    xx_max = 2.5
    xx = np.linspace(-2*np.pi, 2*np.pi, int(1e4))
    width = xx.max()/xx_max

    # plot filters
    plt.figure()
    _, filtr = get_filter(xx, width, choice="LowPass")
    plt.plot(xx/width, filtr, label="LowPass") #LowPass
    _, filtr = get_filter(xx, width, choice="Box")*width
    plt.plot(xx/width, filtr, linestyle="--", label="Box") # Box
    _, filtr = get_filter(xx, width, choice="Gaussian")*width
    plt.plot(xx/width, filtr, linestyle="-.", label="Gaussian") # Gaussian
    plt.xlabel("Dimensioneless length " + r"$x/\lambda$")
    plt.ylabel("Dimensioneless filter " + r"$\lambda G(x)$")
    plt.title("LES filters")
    plt.legend()
    plt.tight_layout()
    plt.show()
