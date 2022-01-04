"""
A pytorch-friendly implementation of the SciPy _ivp OdeSolver and DenseOutput
classes :
    https://github.com/scipy/scipy/blob/v1.7.0/scipy/integrate/_ivp/base.py
    
The structures follows the SciPy implementation with minor changes.
"""

import torch


def check_arguments(fun, y0):
    """Helper function for checking arguments common to all solvers."""
    if not isinstance(y0, torch.Tensor):
        raise TypeError("y0 must be a pytorch Tensor.")
    if y0.dtype != torch.get_default_dtype():
        y0 = y0.to(dtype=torch.get_default_dtype())

    def fun_wrapped(t, y):
        return fun(t, y)

    return fun_wrapped, y0



class OdeSolver(torch.nn.Module):
    """Base class for ODE solvers.
    In order to implement a new solver you need to follow the guidelines:
        1. A constructor must accept parameters presented in the base class
           (listed below) along with any other parameters specific to a solver.
        2. A constructor must accept arbitrary extraneous arguments
           ``**extraneous``, but warn that these arguments are irrelevant
           using `common.warn_extraneous` function. Do not pass these
           arguments to the base class.
        3. A solver must implement a private method `_step_impl(self)` which
           propagates a solver one step further. It must return tuple
           ``(success, message)``, where ``success`` is a boolean indicating
           whether a step was successful, and ``message`` is a string
           containing description of a failure if a step failed or None
           otherwise.
        4. A solver must implement a private method `_dense_output_impl(self)`,
           which returns a `DenseOutput` object covering the last successful
           step.
        5. A solver must have attributes listed below in Attributes section.
           Note that ``t_old`` and ``step_size`` are updated automatically.
        6. Use `fun(self, t, y)` method for the system rhs evaluation, this
           way the number of function evaluations (`nfev`) will be tracked
           automatically.

    Parameters
    ----------
    fun : callable
        Right-hand side of the system. The calling signature is ``fun(t, y)``.
        Here ``t`` is a scalar and the tensor ``y`` has shape (B, neqs, n_points),
        where B denotes the batch size, neqs the number of equations and n_points
        the spatial discretization. ``fun`` must return array_like with shape
        (B, neqs, n_points).
    t0 : float
        Initial time.
    y0 : tensor_like, shape (B, neqs, n_points)
        Initial state.
    t_bound : float
        Boundary time --- the integration won't continue beyond it. It also
        determines the direction of the integration.

    Attributes
    ----------
    n : int
        Number of batchs
    status : string
        Current status of the solver: 'running', 'finished' or 'failed'.
    t_bound : float
        Boundary time.
    direction : float
        Integration direction: +1 or -1.
    t : float
        Current time.
    y : ndarray
        Current state.
    t_old : float
        Previous time. None if no steps were made yet.
    step_size : float
        Size of the last successful step. None if no steps were made yet.
    nfev : int
        Number of the system's rhs evaluations.
    """

    TOO_SMALL_STEP = "Required step size is less than spacing between numbers."

    def __init__(self, fun, t0, y0, t_bound):
        super().__init__()
        self.t_old = None
        self.t = t0
        self._fun, self.y = check_arguments(fun, y0)
        self.t_bound = t_bound
        self.nfev = 0
        self.vectorized = True #TODO: how to handle pointwise model?

        # track the number of rhs evaluations
        def fun(t, y):
            self.nfev += 1
            return self._fun(t, y)
        self.fun = fun

        self.direction = 1
        self.n = self.y.shape[0]
        self.status = 'running'


    @property
    def step_size(self):
        if self.t_old is None:
            return None
        else:
            return abs(self.t - self.t_old)


    def step(self):
        """Perform one integration step.
        Returns
        -------
        message : string or None
            Report from the solver. Typically a reason for a failure if
            `self.status` is 'failed' after the step was taken or None
            otherwise.
        """
        if self.status != 'running':
            raise RuntimeError("Attempt to step on a failed or finished "
                               "solver.")

        if self.n == 0 or self.t >= self.t_bound:
            # Handle corner cases of empty solver or no integration.
            self.t_old = self.t
            self.t = self.t_bound
            message = None
            self.status = 'finished'
        else:
            t = self.t
            success, message = self._step_impl()

            if not success:
                self.status = 'failed'
            else:
                self.t_old = t
                if self.t >= self.t_bound:
                    self.status = 'finished'

        return message


    def dense_output(self):
        """Compute a local interpolant over the last successful step.
        Returns
        -------
        sol : `DenseOutput`
            Local interpolant over the last successful step.
        """
        if self.t_old is None:
            raise RuntimeError("Dense output is available after a successful "
                               "step was made.")

        if self.n == 0 or self.t == self.t_old:
            # Handle corner cases of empty solver and no integration.
            return self.y
        else:
            return self._dense_output_impl()

    def _step_impl(self):
        raise NotImplementedError

    def _dense_output_impl(self):
        raise NotImplementedError


class DenseOutput:
    """Base class for local interpolant over step made by an ODE solver.
    It interpolates between `t_min` and `t_max` (see Attributes below).
    Evaluation outside this interval is not forbidden, but the accuracy is not
    guaranteed.

    Attributes
    ----------
    t_min, t_max : float
        Time range of the interpolation.
    """
    def __init__(self, t_min, t_max):
        self.t_min = t_min
        self.t_max = t_max

    def __call__(self, t):
        """Evaluate the interpolant.
        Parameters
        ----------
        t : float
            Point to evaluate the solution at.
        Returns
        -------
        y : ndarray, shape (B, neqs, n_points)
            Computed values at time t.
        """

        return self._call_impl(t)

    def _call_impl(self, t):
        raise NotImplementedError
