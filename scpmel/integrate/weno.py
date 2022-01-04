"""
Pytorch implementation of the WENO-Z (Weighted Essentially Non-Oscillatory) scheme.
"""

import torch


__all__ = ["WENO_Z", "CircularPad", "CircularConv1d"]


class CircularPad(torch.nn.Module):

    def __init__(self, pad):
        super().__init__()
        self.pad = pad
        
    def forward(self, inputs):
        inputs = torch.nn.functional.pad(inputs, self.pad, mode="circular")
        return inputs


class _Pow(torch.nn.Module):

    def __init__(self, power):
        super().__init__()
        if power is None: # the exponent becomes a trainable parameter
            self.power = torch.nn.Parameter(torch.tensor(power))
        else:
            self.power = power
        
    def forward(self, inputs):
        return torch.pow(inputs, exponent=self.power)


class _Abs(torch.nn.Module):

    def __init__(self):
        super().__init__()
        
    def forward(self, inputs):
        return torch.abs(inputs)


class _Add(torch.nn.Module):

    def __init__(self, const):
        super().__init__()
        self.const = const
        
    def forward(self, inputs):
        return self.const + inputs


class _Mul(torch.nn.Module):

    def __init__(self, const):
        super().__init__()
        self.const = const
        
    def forward(self, inputs):
        return self.const * inputs


class _CustomDiv(torch.nn.Module):

    def __init__(self, eps):
        super().__init__()
        self.eps = eps
        
    def forward(self, inputs):
        num, denom = torch.split(inputs, [1,3], dim=1)
        return torch.div(num, denom+self.eps)


class CircularConv1d(torch.nn.Module):
    """
    Convolutional layer with periodic boundaries and optional fixed weights.

    Parameters
    ----------

    in_channels : int
        The number of input channels.
    out_channels : int
        The number of output channels.
    kernel_size : int or tuple of ints
        The length(s) of the convolution kernel.
    padding : int (default None)
        The circular padding. By default, the padding is defined as
        the integer lower or equal to half the kernel size.
    coeffs : list (default None)
        A list containing the convolution weights.
    grad : bool (default False)
        Fixed (False) or trainable (True) coefficients.
    bias : bool (default False)
        Whether to add a bias or not.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, padding=None,
                 coeffs=None, grad=False, bias=False, **kwargs):
        super().__init__()
        # handle inputs
        if padding is None:
            padding = tuple([kernel_size // 2]*2)
        # build convolution layer
        conv = torch.nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=0, bias=bias, **kwargs)
        if coeffs is not None:
            conv.weight = torch.nn.Parameter(
            torch.tensor(coeffs), requires_grad=grad)
        # build layer
        if not any(padding): # that is if no padding is required
            self.layer = conv
        else:
            self.layer = torch.nn.Sequential(CircularPad(padding), conv)

    def forward(self, inputs):
        return self.layer(inputs)


class WENO_Z(torch.nn.Module):
    """
    Implementation of the WENO-Z scheme [1]_ using the pytorch framework. The
    non-linear WENO-Z weights are computed as in
    (Borges et al. 2008)[https://doi.org/10.1016/j.jcp.2007.11.038]
    with the default power equals to one.

    Parameters
    ----------

    flux_p : boolean
        Wheter to use left or right stencils given the sign of the
        flux acceleration.
    w_pow : int (default 1)
        The WENO-Z weights power parameter that increase the difference of
        scales of distinct weights at non-smooth parts of the solution.
    eps : float (default 1e-40)
        The parameter used to avoid division by zero in fraction denominator.
    DS : bool (default False)
        Enhanced WENO scheme with tunable smoothness indicators, see
        [2]_ for further details.

    References
    ----------
    .. [1] R. Borges, M. Carmona, B. Costa, W. S. Don, "An improved weighted 
    essentially non-oscillatory scheme for hyperbolic conservation laws", J.
    Comput. Phys., Vol. 227, pp. 3191–3211, 2008.
    .. [2] T. Kossaczká, M. Ehrhardt, M. Günther, "Enhanced fifth order WENO Shock-
    Capturing Schemes with Deep Learning", Preprint:arXiv:2103.04988, 2021.
    """    
    
    def __init__(self, flux_p, w_pow=1, eps=1e-13, DS=False):
        super().__init__()
        self._flux_p = flux_p
        self._w_pow = w_pow
        self._eps = eps
        self._DS = DS
        # smoothness and Lagrangian coefficients as convolutional layers
        self.IS = self._smooth_indicators()
        self.LP = self._lagrangian_coeffs()
        # the higher smoothness indicator tau_5
        self.T5 = self._tau5()
        # the un-normalized weights
        self.W = self._unnorm_w()
        # optional enhanced smoothness coefficients
        if self._DS:
            self.DS = self._DS_indicators()
        else:
            self.DS = None


    def forward(self, inputs):
        # smoothness indicators
        IS = self.IS(inputs)
        if self._DS: # enhanced version
            IS *= self.DS(inputs)
        # tau_5 indicators
        T5 = self.T5(IS)
        # un-normalized WENO-Z weights
        W = self.W(torch.cat([T5,IS], dim=1))
        # normalized WENO-Z weights
        W /= W.sum(dim=1, keepdim=True)
        # Lagrangian interpolation
        P = self.LP(inputs)

        return (W*P).sum(dim=1, keepdim=True)


    def _smooth_indicators(self):
        """
        Smoothness indicators build as a convolutional layer.
        """
        # sub-stencils for high-order polynomial approximation
        substencils = torch.tensor(
            [[[1, -2, 1, 0, 0]], [[1, -4, 3, 0, 0]],
             [[0, 1, -2, 1, 0]], [[0, 1, 0, -1, 0]],
             [[0, 0, 1, -2, 1]], [[0, 0, 3, -4, 1]]],
            dtype=torch.get_default_dtype())
        padding = (3,2)
        if not self._flux_p: # flip everything!
            substencils = torch.flip(substencils, dims=(-1,))
            padding = padding[::-1]
        # first convolution layer
        IS_conv1 = CircularConv1d(1, 6, 5, padding, substencils.tolist())
        # second convolution layer
        weights = [[[13./12.], [0.25]]]*3
        IS_conv2 = CircularConv1d(6, 3, 1, (0,0), weights, groups=3)
        # build whole layer
        IS_layer = torch.nn.Sequential(IS_conv1, _Pow(2.0), IS_conv2)

        return IS_layer


    def _lagrangian_coeffs(self):
        """
        Lagrangian interpolation coefficients.
        """
        lcoeffs = torch.tensor(
            [[[2, -7, 11, 0, 0]], [[0, -1, 5, 2, 0]], [[0, 0, 2, 5, -1]]],
            dtype=torch.get_default_dtype())
        padding = (3,2)
        if not self._flux_p: # flip everything!
            lcoeffs = torch.flip(lcoeffs, dims=(-1,))
            padding = padding[::-1]
        # build whole layer
        LP_layer = torch.nn.Sequential(
            CircularConv1d(1, 3, 5, padding, lcoeffs.tolist()),
            _Mul(1./6))
        return LP_layer


    def _tau5(self):
        """
        WENO-Z higher smoothness indicators tau_5 (denoted here T5).
        """
        T5_coeffs = [[[-1.], [0.], [1.]]]
        T5_conv1 = CircularConv1d(3, 1, 1, (0,0), T5_coeffs)
        T5_layer = torch.nn.Sequential(T5_conv1, _Abs())
        return T5_layer


    def _unnorm_w(self):
        """
        WENO-Z un-normalized weights.
        """
        # the ideal weights of the central upwind scheme.
        d_coeffs = [[[1./10]], [[6./10]], [[3./10]]]
        W_conv = CircularConv1d(3, 3, 1, (0,0), d_coeffs, groups=3)
        W_layer = torch.nn.Sequential(
            _CustomDiv(self._eps), _Pow(self._w_pow), _Add(1.0), W_conv)
        return W_layer


    def _DS_indicators(self):
        """
        Optional WENO-DS enhanced smoothness indicators.
        """
        # central finite difference schemes as inputs
        coeffs = [[[1/12., -2/3., 0., 2/3., -1/12.]],
                  [[-1/12., 4/3., -5/2., 4/3., -1/12.]]]
        inputs = CircularConv1d(1, 2, 3, (2,2), coeffs)
        # fist layer
        layer1 = CircularConv1d(2, 3, 5, (2,2), None, bias=True)
        activ1 = torch.nn.ELU()
        # second layer
        layer2 = CircularConv1d(3, 3, 5, (2,2), None, bias=True)
        activ2 = torch.nn.ELU()
        # third layer
        layer3 = CircularConv1d(3, 1, 1, (0,0), None, bias=True)
        activ3 = torch.nn.Sigmoid()
        # final layer
        # sub-stencils traducing the convention of Eqs.(29-30) in [2]_
        substencil = torch.tensor(
            [[[1, 0, 0]], [[0, 1, 0]], [[0, 0, 1]]],
            dtype=torch.get_default_dtype())
        padding = (2,1)
        if not self._flux_p: # flip everything!
            substencil = torch.flip(substencil, dims=(-1,))
            padding = padding[::-1]
        conv_map = CircularConv1d(1, 3, 3, padding, substencil.tolist())
        layer4 = torch.nn.Sequential(conv_map, _Add(0.1)) # C=0.1 see Eq.(28) in [2]_
        # build whole net
        DS_layer = torch.nn.Sequential(
            inputs, layer1, activ1, layer2, activ2, layer3, activ3, layer4)
        return DS_layer
