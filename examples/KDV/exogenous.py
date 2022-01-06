"""
@author: Laurent Gilquin

This file defines the exogenous model of the KDV equation (the enhanced SGS Smagorinsky model).
"""

# global import
import inspect
import math
import numpy as np
import torch
# local import
from scpmel.les import get_filter, get_stencil
from scpmel.training import ConvLayer


class AffineLayer(torch.nn.Module):
    def __init__(self, bounds):
        super().__init__()
        self.bounds = bounds

    def forward(self, inputs):
        return (self.bounds[1]-self.bounds[0]) * inputs + self.bounds[0]


class BasicBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 bias=False, activation='ReLU'):
        super().__init__()
        # get activation function
        activ_func = getattr(torch.nn, activation)
        if "inplace" in inspect.signature(activ_func).parameters:
            activ_layer = activ_func(inplace=True)
        else:
            activ_layer = activ_func()
        self.block = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels, out_channels, kernel_size, bias=bias,
                padding=int(kernel_size//2), padding_mode='circular'),
            activ_layer)

    def forward(self, x):
        return self.block(x)


class BasicNet(torch.nn.Module):
    """
    A basic neural network built as a sequence of blocks with:
        * 1D convolution, odd kernel size and circular padding,
        * activation (inplace if possible),
        * no batch normalization,
        * no skip connection.

    Parameters
    ----------

    in_channels : int
        The number of input channels
    mid_channels : list of ints
        The number of hidden channels per layer.
    out_channels : int
        The number of output channels.
    kernel_size : int
        The filter kernel size.
    bias : boolean
        Wheter to include bias or not in the convolution layer.
    activation : string or list of strings
        A string compatible with the list of torch.nn.activations
        function:
        https://pytorch.org/docs/stable/nn.html?highlight=activations
        If a list of strings is provided, each element will be used in order
        as activation function of each convolutional layer.
    """

    def __init__(self, in_channels=3, mid_channels=[4,4],
                 out_channels=1, kernel_size=5, bias=True, activation="Identity"):
        super().__init__()
        # define inner blocks
        channels = [in_channels] + mid_channels + [out_channels]
        inner_blocks = []
        if not isinstance(activation, list):
            activation = [activation]*len(channels)
        # append middle blocks
        for idx in range(1, len(channels)-1):
            inner_blocks.extend(
                BasicBlock(channels[idx-1], channels[idx], kernel_size,
                           bias, activation[idx-1]).block
                )
        # append last block with identity layer
        inner_blocks.extend(
                BasicBlock(channels[-2], channels[-1], kernel_size,
                           bias, activation[-1]).block
                )
        self.net = torch.nn.Sequential(*inner_blocks)

    def forward(self, x):
        return self.net(x)


class Smagorinsky(torch.nn.Module):
    """
    Smagorinsky model implemented as a neural network.

    Parameters
    ----------
    cutoff : float
        The filter cutoff length. The associated grid step is twice smaller.
    scale : float
        The spatial domain length.
    acc : int
        The derivative accuracy order.
    model : torch.nn.Module
        The network architecture used for the Smagorinsky constant.
    bounds : tuple or list
        The bounds for the Smagorinsky constant.
    """

    def __init__(self, cutoff, scale, acc, model, bounds=[0.1, 0.3]):

        super().__init__()
        # cutoff length
        self.cutoff = cutoff
        dx = cutoff / 2
        # define LES filter
        xx = np.linspace(0, scale, int(scale // dx), endpoint=False)
        _, weights = get_filter(xx-scale/2, cutoff, "Box", tol=1e-8)
        # check if number of weights is odd
        if len(weights) % 2 == 0:
            weights = np.r_[weights, 0.0] # add a zero to the right
        self.les_filter = ConvLayer(torch.tensor(
            weights, dtype=torch.get_default_dtype()).reshape((1,1)+(len(weights),)),
            scheme="centered")
        # set derivative layer
        self.derivative = ConvLayer(get_stencil(dx, order=1, acc=acc, scheme="center"),
                                    scheme="centered")
        # the nn modeling the Smagorinsky coefficient
        self.Cs = torch.nn.Sequential(
            *model.children(),
            torch.nn.Hardtanh(bounds[0], bounds[1])
            )

    def forward(self, x):
        """
        Forward method to build the differentiation graph.
        """
        # get derivative with detached graph
        xdiff = self.derivative(x)
        # concatenate the three quantities
        x = torch.cat((xdiff, x - self.les_filter(x)), dim=1)
        # get constant field
        x = self.Cs(x)
        # Smagorinsky subgrid tensor
        x = (x * self.cutoff)**2 * torch.abs(xdiff) * xdiff
        return self.derivative(math.sqrt(2.0) * x)

    def predict(self, x):
        """
        Predict method for forecasting.
        """
        self.eval()
        with torch.no_grad():
            x = self.forward(x)
        self.train()
        return x

    def _get_Cs(self, x):
        with torch.no_grad():
            xdiff = self.derivative(x)
            x = torch.cat((xdiff, x - self.les_filter(x)), dim=1)
            return self.Cs(x)


class EnhancedSmagorinsky(Smagorinsky):
    """
    The neural network inspired from:
    [Sarghini 2003] (https://doi.org/10.1016/S0045-7930(01)00098-6)
    used to estimate the Smagorinsky constant Cs. The implementation used here
    rewrite the original one through convolutional layers.
    """
    def __init__(self, cutoff, scale, acc, bounds=[0., 1.0]):
        model = BasicNet(in_channels=2, mid_channels=[6,3], kernel_size=5,
                         bias=True, activation=["ELU", "ELU", "ELU"])
        super().__init__(cutoff, scale, acc, model, bounds)


if __name__ == '__main__':

    import torchinfo
    torch.set_default_dtype(torch.float64)
    # custom config
    scale = 500
    npts = 4096
    cutoff = scale / 128
    acc = 2
    # test Shargini model
    TauNet = EnhancedSmagorinsky(cutoff, scale, acc)
    torchinfo.summary(TauNet, (10,1,100), verbose=1, dtypes=[torch.get_default_dtype()])
