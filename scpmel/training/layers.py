#global imports
import torch
import torch.nn.functional as F

__all__ = ["ConvLayer", "PeriodicLayer"]


class PeriodicLayer(torch.nn.Module):
    """ Periodization of the grid in accordance with filter size"""

    def __init__(self, kernel_size, scheme):
        super().__init__()
        self.kernel_size = kernel_size
        self.scheme = scheme

        # check that all kernel sizes are odd
        if torch.any(torch.fmod(self.kernel_size, 2) != 1):
            raise ValueError(f"kernel_size {kernel_size} is not **odd** integer")
            
        # check correct scheme name
        if self.scheme not in ["centered", "upwind", "backwind"]:
            raise NameError(
                """Scheme argument must be one of {'centered', 'upwind, 
                backwind'}.""")
        # get padding values
        self.pad = tuple(self._get_pad().int().tolist())

    def _get_pad(self):
        """get correct pad according to scheme."""
        pad_len = torch.div(self.kernel_size, 2, rounding_mode='trunc')
        if self.scheme == "centered":
            pad = pad_len.repeat_interleave(2)   
        else:
            pad_len *= 2 # single direction
            pad = pad_len.repeat_interleave(2)
            if self.scheme == "upwind":
                pad[::2] = 0 # zero even indexes
            elif self.scheme == "backwind":
                pad[1::2] = 0 # zero odd indexes
        return pad

    def get_config(self):
        config = {
            'pad': self.pad,
            'kernel_size': self.kernel_size
        }
        return config

    def forward(self, inputs):
        
        periodic_inputs = F.pad(inputs, self.pad, mode="circular")
        return periodic_inputs


def ConvLayer(kernel, scheme="centered"):
    """
    Function that build a Conv-Net with periodic boundaries.

    Parameters
    ----------
    kernel : torch.Tensor
        A tensor containing the convolutional layer weights.

    Returns
    -------
    The Conv-Net as an instance of torch.nn.Sequential.
    """

    kernel_size = torch.tensor(kernel.shape[2:])
    dimension = len(kernel_size)
    if dimension == 1:
        conv = torch.nn.Conv1d
    else:
        conv = torch.nn.Conv2d

    ConvLayer = conv(1, 1, kernel_size, padding=0, bias=False)
    ConvLayer.weight = torch.nn.Parameter(kernel, requires_grad=False)
    inner_layers = [
        PeriodicLayer(kernel_size, scheme),
        ConvLayer
        ]

    return torch.nn.Sequential(*inner_layers)