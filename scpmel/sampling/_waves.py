# global import
import numpy as np


def pke_wave(dx, start, length):
    """
    Initial condition used in Pannekoucke20.
    """
    x = np.arange(start, length, dx)
    out = 0.25*(1+np.cos(2*np.pi/length*(x-1)))
    return out


def sin_wave(xm, start=10, end=50):
    """
    Set a sinus wave as initial condition.
    """
    out = np.sin(2*np.pi*xm/(end - start))

    return out    


def square_wave(xm, start=10, end=50, altitude=1):
    """
    Set a square wave as initial condition.
    """
    out = np.zeros(xm)
    out[start:end] += altitude
    return out


def gaussian_wave(xm, center=0.0, scale=1.0, norm=False):
    """
    Set a gaussian wave as initial condition.
    """

    # build the wave
    out = np.exp(-1/2*((xm-center)/scale)**2)
    if norm:
        out *= 1/(np.sqrt(2*np.pi)*scale)
    return out/out.max()


def multi_waves(dx, xmax, centers, scales):
    """
    Multiple gaussian waves.
    """
    out = np.zeros(int(xmax/dx))
    for center, scale in zip(centers, scales):
        out += gaussian_wave(dx, xmax, center, scale)
    return out/out.max()