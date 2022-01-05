"""
A pytorch-friendly implementation of the SciPy solve_ivp function :
    https://github.com/scipy/scipy/blob/v1.7.0/scipy/integrate/_ivp/ivp.py

The structures follows the SciPy implementation with minor changes (ex: removed
intepolants from solve_ip output, etc ...).
"""

# global import
import inspect
import torch
# relative imports
from ._rk import (HEUNEULER, RK23, SSPRK23, RK5SSP3, RK45_DP, RK2, RK3, SSPRK3, RK4, RK4_38, RK5_3, HEUN2, HEUN3,
                 RALSTON3, RALSTON4, EULER)
from ._base import OdeSolver
from ._common import close_to_any


METHODS = {'HEUNEULER': HEUNEULER, 'RK23': RK23, 'SSPRK23': SSPRK23, 'RK5SSP3': RK5SSP3,
           'RK45_DP': RK45_DP, 'RK2': RK2, 'RK3': RK3, 'SSPRK3': SSPRK3, 'RK4': RK4,
           'RK4_38': RK4_38, 'RK5_3': RK5_3, 'HEUN2': HEUN2, 'HEUN3': HEUN3, 'RALSTON3': RALSTON3,
           'RALSTON4': RALSTON4, 'EULER': EULER
           }

MESSAGES = {0: "The solver successfully reached the end of the integration interval.",
            1: "A termination event occurred."}

class IVP(object):
    """
    Interface class for instantiating a time setup and solving an Initial
    Value Problem (IVP). Useful for calling the forward method of a torch.nn.Module
    with a pre-instantiated time setup as it only requires to pass the inputs.
    """

    def __init__(self):
        self.dt = None
        self.method = None
        self.t_span = None
        self.t_eval = None
        self.dense_output = None
        self.device = "cpu"

    def set_time(self, dt, t_span, t_eval, method, dense_output, **options):
        """
        Set the time parameters: time step, the starting and ending integration
        times, the time scheme, the times at which the solution should be extracted
        and wheter to use o local interpolant or not.

        Parameters
        ----------
        dt : float.
            The time step.

        t_span : tuple
            Starting and ending points of the time integration.

        t_eval : int
            The step times at which the solution should be extracted.

        method : str
            The name of the time scheme to be used.

        dense_output : bool (default false)
            if True, local interpolant is used to evaluate the state at t_eval,
            otherwise returns only the points selected by the solver matching
            t_eval.
            

        Returns
        -------
        None.
        """
        self._dt = dt
        self.t_span = t_span
        self.t_eval = t_eval
        self._set_time_scheme(method)
        self.dense_output = dense_output
        self._options = options
        return None

    def set_device(self, device):
        """
        Set the torch device on which to carry out the simulations, defaults to "cpu".
        """
        self.device = device

    def _set_time_scheme(self, time_scheme:str):
        if time_scheme in METHODS.keys():
            self.time_scheme = time_scheme
        else:
            raise ValueError(f"""Time scheme {time_scheme} is not valid, it must \
                             be one of {METHODS}.""")

    def forecast(self, fun, y0, full=False):
        """ 
        Gimmick function to call solve_ivp through the class.
        Parameters
        ----------
        fun : callable
            Right-hand side of the system. The calling signature is ``fun(t, y)``.
            Here `t` is a scalar and the tensor ``y`` has shape (B, neqs, n_points),
            where B denotes the batch size, neqs the number of equations and n_points
            the spatial discretization. ``fun`` must return array_like with shape
            (B, neqs, n_points).
        y0 : tensor, shape (B, neqs, n_points)
            Initial state.
        full : bool (default Fase)
            Whether to return the full output of the solver or only the state(s).
        """
        assert self.time_scheme is not None, """The time setting must be instantiated
        through the setup() method prior to perform predictions."""
        res = solve_ivp(fun, self.t_span, y0, self.time_scheme, self.t_eval,
                        dense_output=self.dense_output, device=self.device,
                        first_step=self._dt, **self._options)
        if full:
            return res
        elif len(res.y) == 1:
            return res.y[0]
        else:
            return res.y[-1]


class OdeResult(dict):
    """ Store the ODE results.
    Attributes
    ----------
    x : ndarray
        The solution of the optimization.
    success : bool
        Whether or not the optimizer exited successfully.
    status : int
        Termination status of the optimizer. Its value depends on the
        underlying solver. Refer to `message` for details.
    message : str
        Description of the cause of the termination.
    nfev: int
        Number of evaluations of the objective function.
    nit : int
        Number of iterations performed by the optimizer.
    Notes
    -----
    There may be additional attributes not listed above depending of the
    specific solver. Since this class is essentially a subclass of dict
    with attribute accessors, one can see which attributes are available
    using the `keys()` method.
    """

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        if self.keys():
            m = max(map(len, list(self.keys()))) + 1
            return '\n'.join([k.rjust(m) + ': ' + repr(v)
                              for k, v in sorted(self.items())])
        else:
            return self.__class__.__name__ + "()"

    def __dir__(self):
        return list(self.keys())


def solve_ivp(fun, t_span, y0, method="RK45_DP", t_eval=None, dense_output=False,
              device='cpu', **options):
    """Solve an initial value problem for a system of ODEs.
    This function numerically integrates a system of ordinary differential
    equations given an initial value::
        dy / dt = f(t, y)
        y(t0) = y0
    Here t is a 1-D independent variable (time), y(t) is an
    n-dimensional vector-valued (n_points) function (state), and an n-dimensional
    vector-valued (n_points) function f(t, y) determines the differential equations.
    The goal is to find y(t) approximately satisfying the differential
    equations, given an initial value y(t0)=y0.
    Parameters
    ----------
    fun : callable
        Right-hand side of the system. The calling signature is ``fun(t, y)``.
        Here `t` is a scalar and the tensor ``y`` has shape (B, neqs, n_points),
        where B denotes the batch size, neqs the number of equations and n_points
        the spatial discretization. ``fun`` must return array_like with shape
        (B, neqs, n_points).
    t_span : 2-tuple of floats
        Interval of integration (t0, tf). The solver starts with t=t0 and
        integrates until it reaches t=tf.
    y0 : tensor, shape (B, neqs, n_points)
        Initial state.
    method : string or `OdeSolver`, optional
        Available adaptative integration methods are:
            * 'RK45': Explicit Runge-Kutta method of order 5(4) [1]_.
              The error is controlled assuming accuracy of the fourth-order
              method, but steps are taken using the fifth-order accurate
              formula (local extrapolation is done). A quartic interpolation
              polynomial is used for the dense output [2]_. Can be applied in
              the complex domain.
            * 'RK23': Explicit Runge-Kutta method of order 3(2) [3]_. The error
              is controlled assuming accuracy of the second-order method, but
              steps are taken using the third-order accurate formula (local
              extrapolation is done). A cubic Hermite polynomial is used for the
              dense output. Can be applied in the complex domain.
        Explicit Runge-Kutta methods should be used for non-stiff problems.
        You can also pass an arbitrary class derived from `OdeSolver` which
        implements the solver.
    t_eval : list or None, optional
        Times at which to store the computed solution, must be sorted and lie
        within `t_span`. If None (default), use points selected by the solver.
    dense_output : bool (default false)
        if True, local interpolant is used to evaluate the state at t_eval,
        otherwise returns only the points selected by the solver matching
        t_eval.
    device : torch.device
        The desired device of returned tensor. Device will be the CPU for CPU
        tensor types and the current CUDA device for CUDA tensor types.
    options
        Options passed to a chosen solver. All options available for already
        implemented solvers are listed below.
        * first_step : float or None, optional
            Initial step size. Default is `None` which means that the algorithm
            should choose.
        * max_step : float, optional
            Maximum allowed step size. Default is np.inf, i.e., the step size is not
            bounded and determined solely by the solver.
        * rtol, atol : float or array_like, optional
            Relative and absolute tolerances. The solver keeps the local error
            estimates less than ``atol + rtol * abs(y)``. Here `rtol` controls a
            relative accuracy (number of correct digits). But if a component of `y`
            is approximately below `atol`, the error only needs to fall within
            the same `atol` threshold, and the number of correct digits is not
            guaranteed. If components of y have different scales, it might be
            beneficial to set different `atol` values for different components by
            passing array_like with shape (n,) for `atol`. Default values are
            1e-3 for `rtol` and 1e-6 for `atol`.
    Returns
    -------
    Bunch object with the following fields defined:
    t : ndarray, shape (n_points,)
        Time points.
    y : ndarray, shape (B, neqs, n_points)
        Values of the solution at `t`.
    nfev : int
        Number of evaluations of the right-hand side.
    status : int
        Reason for algorithm termination:
            * -1: Integration step failed.
            *  0: The solver successfully reached the end of `tspan`.
            *  1: A termination event occurred.
    message : string
        Human-readable description of the termination reason.
    success : bool
        True if the solver reached the interval end or a termination event
        occurred (``status >= 0``).
    References
    ----------
    .. [1] J. R. Dormand, P. J. Prince, "A family of embedded Runge-Kutta
           formulae", Journal of Computational and Applied Mathematics, Vol. 6,
           No. 1, pp. 19-26, 1980.
    .. [2] L. W. Shampine, "Some Practical Runge-Kutta Formulas", Mathematics
           of Computation,, Vol. 46, No. 173, pp. 135-150, 1986.
    .. [3] P. Bogacki, L.F. Shampine, "A 3(2) Pair of Runge-Kutta Formulas",
           Appl. Math. Lett. Vol. 2, No. 4. pp. 321-325, 1989.
    """
    if method not in METHODS and not (
            inspect.isclass(method) and issubclass(method, OdeSolver)):
        raise ValueError("`method` must be one of {} or OdeSolver class."
                         .format(METHODS))
    elif method in METHODS:
        methodf = METHODS[method]

    t0, tf = float(t_span[0]), float(t_span[1])

    if t_eval is not None and not torch.is_tensor(t_eval):
        t_eval = torch.tensor(t_eval, device=device)
        
        if t_eval.ndim != 1:
            raise ValueError("`t_eval` must be 1-dimensional.")

        if torch.any(t_eval < min(t0, tf)) or torch.any(t_eval > max(t0, tf)):
            raise ValueError("Values in `t_eval` are not within `t_span`.")

        d = torch.diff(t_eval)
        if torch.any(d <= 0) :
            raise ValueError("Values in `t_eval` are not properly sorted.")

    t_eval_i = 0

    solver = methodf(fun, t0, y0, tf, **options)
    solver.to(device=device, dtype=torch.get_default_dtype())

    if t_eval is None:
        ts = [t0]
        ti = None
        ys = [y0]
    elif not dense_output:
        ts = [t0]
        ti = []
        ys = [y0]
    else:
        ts = []
        ti = []
        ys = []

    status = None
    while status is None:
        message = solver.step()

        if solver.status == 'finished':
            status = 0
        elif solver.status == 'failed':
            status = -1
            break

        t = solver.t
        ts.append(t)
        y = solver.y

        if t_eval is None:
            ys.append(y)
        elif dense_output:
            #TODO: implement a local interpolant for all methods so that no
            # NotImplementedError get throw
            sol = solver.dense_output()
            # The value in t_eval equal to t will be included.
            t_eval_i_new = torch.searchsorted(t_eval, t, right=True).item()
            t_eval_step = t_eval[t_eval_i:t_eval_i_new]
            if t_eval_step.size()[0] > 0:
                ti += t_eval_step.tolist()
                ys += sol(t_eval_step)
                t_eval_i = t_eval_i_new
        elif not dense_output and close_to_any(t, t_eval, device):
            ti.append(t)
            ys.append(y)

    message = MESSAGES.get(status, message)

    return OdeResult(t=ts, ti=ti, y=ys, nfev=solver.nfev, status=status,
                     message=message, success=status >= 0)