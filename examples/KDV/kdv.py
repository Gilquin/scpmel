"""
@author: Laurent Gilquin

Pytorch implementation of the time-space discretization of the 1D Korteweg–De Vries (KDV)
equation. For further details about the KDV equation, see:
    * https://en.wikipedia.org/wiki/Korteweg%E2%80%93De_Vries_equation,
    * https://www.researchgate.net/publication/242478002_An_introduction_to_internal_waves , Chapter 8.2
    
The space discretization is performed via the WENO-Z scheme:
    https://www.sciencedirect.com/science/article/pii/S0021999110006431?via%3Dihub
with the Lax-Friedrichs flux splitting method.
"""


import torch
import torch.utils.checkpoint as checkpoint
from scpmel.integrate import IVP, WENO_Z, CircularPad

class KDV(torch.nn.Module, IVP):

    # Prognostic functions
    prognostic_functions = (
        'u', # Write comments on the function here
        )

    # Exogenous functions
    exogenous_functions = (
        'tau', # Write comments on the exogenous function here
        )

    # Spatial coordinates
    coordinates = (
        'x', # Write comments on the coordinate here
        )

    # Set constants
    constants = (
        'A', # Writes comment on the constant here
        'B', # Writes comment on the constant here
        'c0', # Writes comment on the constant here
        'nu', # Writes comment on the constant here
        )


    def __init__(self, shape, lengths, **kwargs):

        super().__init__()
        IVP.__init__(self)

        #---------------------------------
        # Set index array from coordinates
        #---------------------------------

        # set shape
        if len(shape)!=len(self.coordinates):
            raise ValueError(f"len(shape) {len(shape)} is different from len(coordinates) {len(self.coordinates)}")
        else:
            self.shape = shape

        # set input shape for coordinates
        self.input_shape_x = shape[0]
        # set lengths
        if len(lengths)!=len(self.coordinates):
            raise ValueError(f"len(lengths) {len(lengths)} is different from len(coordinates) {len(self.coordinates)}")
        else:
            self.lengths = lengths

        # Set spatial steps and domain
        self.dx = tuple(length/shape for length, shape in zip(self.lengths, self.shape))

        #-------------------------------
        # Set dictionaries
        #-------------------------------
        self.inputs_dict = {
            'u': 0,
            }
        self.outputs_dict = {
            'trend_u': 0,
            }

        #-------------------------------
        # Set exogenous functions
        #-------------------------------
        # Set a default value for exogenous functions
        self.tau = None

        #-------------------------------
        # Set constants of the model
        #-------------------------------
        # Set a default value for constants
        self.A = None
        self.B = None
        self.c0 = None
        self.nu = None

        # Set constant values from external **kwargs (when provided)
        for key in kwargs:
            if key in self.constants:
                setattr(self, key, kwargs[key])

        # Alert when a constant is None
        for constant in self.constants:
            if getattr(self, constant) is None:
                print(f"Warning: constant `{constant}` has to be set")

        #-------------------------------
        # Instantiate the derivatives
        #-------------------------------
        kernel_Du_x_o1 = torch.tensor([
            -1/(self.dx[self.coordinates.index('x')]),
            1/(self.dx[self.coordinates.index('x')])],
         dtype=torch.get_default_dtype()).reshape((1,1)+(2,))
        conv_x_o1 = torch.nn.Conv1d(1, 1, 2, padding=0, bias=False)
        conv_x_o1.weight = torch.nn.Parameter(kernel_Du_x_o1, requires_grad=False)
        self.Du_x_o1 = torch.nn.Sequential(CircularPad((1,1)), conv_x_o1)

        kernel_Du_x_o3 = torch.tensor([
        1/(8*self.dx[self.coordinates.index('x')]**3),
        -1/self.dx[self.coordinates.index('x')]**3,
        13/(8*self.dx[self.coordinates.index('x')]**3),
        0.0,
        -13/(8*self.dx[self.coordinates.index('x')]**3),
        self.dx[self.coordinates.index('x')]**(-3),
        -1/(8*self.dx[self.coordinates.index('x')]**3)],
            dtype=torch.get_default_dtype()).reshape((1,1)+(7,))
        conv_x_o3 = torch.nn.Conv1d(1, 1, 7, padding=0, bias=False)
        conv_x_o3.weight = torch.nn.Parameter(kernel_Du_x_o3, requires_grad=False)
        self.Du_x_o3 = torch.nn.Sequential(CircularPad((3,3)), conv_x_o3)

        #-------------------------------
        # Instantiate WENO-Z model
        #-------------------------------
        weno_keys = {key: value for key,value in kwargs.items()\
                     if key in ['w_pow', 'eps', 'DS']}
        self.weno_p = WENO_Z(flux_p=True, **weno_keys)
        self.weno_n = WENO_Z(flux_p=False, **weno_keys)

        #-------------------------------
        # Finite Volume to Finite Difference (interface to cell values)
        #-------------------------------
        kernel_cell = torch.tensor([-1, 1],
         dtype=torch.get_default_dtype()).reshape((1,1)+(2,))
        self.conv_cell = torch.nn.Conv1d(1, 1, 2, padding=0, bias=False)
        self.conv_cell.weight = torch.nn.Parameter(kernel_cell, requires_grad=False)


    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.set_device(*args, **kwargs)
        return self


    def make_exogenous(self, functions):
        """
        Instantiate exogenous (or constant) functions.

        Parameters
        ----------
        functions : dict.
            A dict of variables to be replaced by callables.

        Returns
        -------
        None.
        """

        exogenous_keys = ['tau']
        constant_keys = []
        for fname, func in functions.items():
            if (fname not in exogenous_keys) and (fname not in constant_keys):
                raise KeyError(
                    f"{fname} is not a valid exogenous or constant function name."
                    )
            elif not callable(func):
                raise TypeError(
                    f"{fname} object is not callable."
                    )
            else:
                setattr(self, fname, func)
        return None



    def forward(self, inputs):
        """
        Forward method to train the model.

        Parameters
        ----------
        inputs : torch.Tensor
            A torch Tensor containing the inputs stacked along the first
          dimension.

        Returns
        -------
        A torch.tensor containing the model predictions.
        """

        # check that the time configuration has been instantiated
        assert self.time_scheme is not None, """The time setting must be instantiated
        through the set_time() method prior to perform predictions."""

        # predict through the time scheme
        self._checkpointing = False
        self.train()
        output = self.forecast(self.trend, inputs, full=False)
        return output


    def predict(self, inputs):
        """
        Predict method to forecast.

        Parameters
        ----------
        inputs : torch.Tensor
            A torch Tensor containing the inputs stacked along the first
          dimension.

        Returns
        -------
        A torch.tensor containing the model predictionss.
        """

        # check that the time configuration has been instantiated
        assert self.time_scheme is not None, """The time setting must be instantiated
        before making predictions through the set_time() method."""

        # predict through the time scheme
        self._checkpointing = False
        self.eval()
        with torch.no_grad():
            output = self.forecast(self.trend, inputs, full=True)
        return output


    def trend(self, t, inputs):
        """
        Evaluates the model trend at time t.

        Parameters
        ----------
        inputs : torch.Tensor
            A torch Tensor containing the inputs stacked along the first
            dimension.

        Returns
        -------
        A torch.tensor containing the model trend.
        """
        # exogenous functions
        if not self._checkpointing:
            tau = self.tau(inputs)
        else:
            tau = checkpoint.checkpoint(self.custom(self.tau), inputs.requires_grad_())
        # Computation of trend_u
        trend_u = - self._lax_friedrichs(t, inputs)/self.dx[self.coordinates.index('x')] +\
            tau + self._dissipation(t, inputs) - self._dispersion(t, inputs)
        return trend_u


    def _lax_friedrichs(self, t, inputs):
        u_max = torch.max(torch.abs(self.c0 + self.A * inputs))
        f_plus = 1/2 * self.weno_p(self.A/2 * inputs**2 + (self.c0 + u_max) * inputs)
        f_minus = 1/2 * self.weno_n(self.A/2 * inputs**2 + (self.c0 - u_max) * inputs)
        return self.conv_cell(f_plus + f_minus)


    def _dissipation(self, t, inputs):
        Du_x_o1 = self.Du_x_o1(inputs)
        return self.nu * self.conv_cell(Du_x_o1)


    def _dispersion(self, t, inputs):
        return self.B * self.Du_x_o3(inputs)


    def custom(self, module):
        """ Used to enable checkpointing a torch.nn.Module."""
        def custom_forward(*inputs):
            inputs = module(inputs[0])
            return inputs
        return custom_forward
