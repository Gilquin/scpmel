"""
Created on Tue Jul 27 11:29:32 2021

A pytorch-friendly implementation of the SciPy Runge-Kutta methods and some others:
    https://github.com/scipy/scipy/blob/v1.7.0/scipy/integrate/_ivp/rk.py
The structures (classes, methods, ...) follows the SciPy implementation.

@author: Laurent Gilquin
"""

import torch
from .base import OdeSolver, DenseOutput
from .common import (validate_max_step, validate_tol, select_initial_step,
                     norm, warn_extraneous, validate_first_step)
from .common import EPS, FMAX


SAFETY = 0.9 # multiplicative factor for steps computed from asymptotic behaviour of errors
MIN_FACTOR = 0.2  # minimum allowed decrease in a step size.
MAX_FACTOR = 10  # maximum allowed increase in a step size.

def rk_step(fun, t, y, f, h, A, B, C, K):
    """Perform a single Runge-Kutta step.
    This function computes a prediction of an explicit Runge-Kutta method and
    also estimates the error of a less accurate method.
    Notation for Butcher tableau is as in [1]_.
    Parameters
    ----------
    fun : callable
        Right-hand side of the system.
    t : float
        Current time.
    y : tensor, shape (B, neqs, n_points)
        Current state.
    f : tensor, shape (B, neqs, n_points)
        Current value of the derivative.
    h : float
        Step to use.
    A : tensor, shape (n_stages, n_stages)
        Coefficients for combining previous RK stages to compute the next
        stage. For explicit methods the coefficients at and above the main
        diagonal are zeros.
    B : tensor, shape (n_stages)
        Coefficients for combining RK stages for computing the final
        prediction.
    C : tensor, shape (n_stages)
        Coefficients for incrementing time for consecutive RK stages.
        The value for the first stage is always zero.
    K : tensor, shape (B, neqs, n_points, n_stages + 1)
        Storage array for putting RK stages here. The last stage is a linear
        combination of the previous stages with coefficients of B.
    Returns
    -------
    y_new : tensor, shape (B, neqs, n_points)
        Solution at t + h computed with a higher accuracy.
    f_new : tensor, shape (B, neqs, n_points)
        Derivative ``fun(t + h, y_new)``.
    References
    ----------
    .. [1] E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential
           Equations I: Nonstiff Problems", Sec. II.4.
    """
    K[...,0] = f
    for s, (a, c) in enumerate(zip(A[1:], C[1:]), start=1):
        dy = h * K[...,:s] @ a[:s]
        K[...,s] = fun(t + c * h, y + dy)

    y_new = y + h * K[...,:-1] @ B
    f_new = fun(t + h, y_new)

    K[...,-1] = f_new

    return y_new, f_new


class AdaptativeRungeKutta(OdeSolver):
    """Base class for explicit adaptative Runge-Kutta methods.
    """
    C: torch.Tensor = NotImplemented
    A: torch.Tensor = NotImplemented
    B: torch.Tensor = NotImplemented
    E: torch.Tensor = NotImplemented
    P: torch.Tensor = NotImplemented
    order: int = NotImplemented
    error_estimator_order: int = NotImplemented
    n_stages: int = NotImplemented

    def __init__(self, fun, t0, y0, t_bound, rtol=1e-3, atol=1e-6, max_step=FMAX,
                 first_step=None, **extraneous):
        warn_extraneous(extraneous)
        super().__init__(fun, t0, y0, t_bound)
        self.y_old = None
        self.min_step = 10 * EPS
        self.max_step = validate_max_step(max_step)
        self.rtol, self.atol = validate_tol(rtol, atol, self.n)
        self.f = self.fun(self.t, self.y)

        if first_step is None:
            self.h = select_initial_step(
                self.fun, self.t, self.y, self.f, 1.0, self.error_estimator_order,
                self.rtol, self.atol)
        else:
            self.h = validate_first_step(first_step, t0, t_bound)
        self.K = torch.empty(y0.shape + (self.n_stages + 1,), dtype=y0.dtype)
        self.error_exponent = -1 / (self.error_estimator_order + 1)

    def _estimate_error(self, K, h):
        return h * K @ self.E

    def _estimate_error_norm(self, K, h, scale):
        return norm(self._estimate_error(K, h) / scale)

    def _step_impl(self):

        max_step = self.max_step
        rtol = self.rtol
        atol = self.atol

        if self.h > max_step:
            h = max_step
        elif self.h < self.min_step:
            h = self.min_step
        else:
            h = self.h

        step_accepted = False
        step_rejected = False

        while not step_accepted:
            if h < self.min_step:
                return False, self.TOO_SMALL_STEP

            t_new = self.t + h
            if t_new > self.t_bound:
                t_new = self.t_bound
            h = t_new - self.t

            y_new, f_new = rk_step(self.fun, self.t, self.y, self.f, h, self.A,
                                   self.B, self.C, self.K)
            scale = atol + torch.max(self.y.abs(), y_new.abs()) * rtol
            error_norm = self._estimate_error_norm(self.K, h, scale)

            if error_norm < 1:
                if error_norm == 0:
                    factor = MAX_FACTOR
                else:
                    factor = min(MAX_FACTOR,
                                 SAFETY * error_norm ** self.error_exponent)

                if step_rejected:
                    factor = min(1, factor)

                h *= factor

                step_accepted = True
            else:
                h *= max(MIN_FACTOR, SAFETY * error_norm ** self.error_exponent)
                step_rejected = True

        self.y_old = self.y

        self.t = t_new
        self.y = y_new

        self.h = h
        self.f = f_new

        return True, None

    def _dense_output_impl(self):
        Q = self.K @ self.P
        return RkDenseOutput(self.t_old, self.t, self.y_old, Q)

    # hacky way of dynamically moving the tensors to the correct device
    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs) 
        self.A = self.A.to(*args, **kwargs)
        self.B = self.B.to(*args, **kwargs)
        self.C = self.C.to(*args, **kwargs)
        self.E = self.E.to(*args, **kwargs)
        self.P = self.P.to(*args, **kwargs)
        self.K = self.K.to(*args, **kwargs)
        return self


class HEUNEULER(AdaptativeRungeKutta):
    """The simplest adaptive Runge–Kutta method involves combining Heun's method
    [1]_, which is order 2, with the Euler method, which is order 1. The dense
    output corresponds to a strong stability preserving second-order
    interpolator [2]_.
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
    y0 : array_like, shape (B, neqs, n_points)
        Initial state.
    t_bound : float
        Boundary time - the integration won't continue beyond it. It also
        determines the direction of the integration.
    first_step : float or None, optional
        Initial step size. Default is ``None`` which means that the algorithm
        should choose.
    max_step : float, optional
        Maximum allowed step size. Default is np.inf, i.e., the step size is not
        bounded and determined solely by the solver.
    rtol, atol : float and array_like, optional
        Relative and absolute tolerances. The solver keeps the local error
        estimates less than ``atol + rtol * abs(y)``. Here, `rtol` controls a
        relative accuracy (number of correct digits). But if a component of `y`
        is approximately below `atol`, the error only needs to fall within
        the same `atol` threshold, and the number of correct digits is not
        guaranteed. If components of y have different scales, it might be
        beneficial to set different `atol` values for different components by
        passing tensor with shape (neqs,) for `atol`. Default values are
        1e-3 for `rtol` and 1e-6 for `atol`.
    Attributes
    ----------
    n : int
        Number of batchs.
    status : string
        Current status of the solver: 'running', 'finished' or 'failed'.
    t_bound : float
        Boundary time.
    direction : float
        Integration direction: +1 or -1.
    t : float
        Current time.
    y : tensor
        Current state.
    t_old : float
        Previous time. None if no steps were made yet.
    step_size : float
        Size of the last successful step. None if no steps were made yet.
    nfev : int
        Number evaluations of the system's right-hand side.
    References
    ----------
    .. [1] E. Süli, D. Mayers, "An Introduction to Numerical Analysis",
    Cambridge University Press, ISBN 0-521-00794-1, 2003.
    .. [2] D.I. Ketcheson, L. Lóczi, A. Jangabylova, A. Kusmanov, "Dense output
    for strong stability preserving Runge-Kutta methods", arXiv:1605.02429v2
    [math.NA] , 2016.
    """
    order = 2
    error_estimator_order = 1
    n_stages = 2
    C = torch.tensor([0., 1.])
    A = torch.tensor([[0., 0.], [1., 0.]])
    B = torch.tensor([1./2, 1./2])
    E = torch.tensor([1./2, -1./2, 0.0])
    P = torch.tensor([[1., -1./2], [0., 1./2], [0., 0.]])


class RK23(AdaptativeRungeKutta):
    """Explicit Runge-Kutta method of order 3(2).
    This uses the Bogacki-Shampine pair of formulas [1]_. The error is controlled
    assuming accuracy of the second-order method, but steps are taken using the
    third-order accurate formula (local extrapolation is done). A cubic Hermite
    polynomial is used for the dense output.
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
    y0 : array_like, shape (B, neqs, n_points)
        Initial state.
    t_bound : float
        Boundary time - the integration won't continue beyond it. It also
        determines the direction of the integration.
    first_step : float or None, optional
        Initial step size. Default is ``None`` which means that the algorithm
        should choose.
    max_step : float, optional
        Maximum allowed step size. Default is np.inf, i.e., the step size is not
        bounded and determined solely by the solver.
    rtol, atol : float and array_like, optional
        Relative and absolute tolerances. The solver keeps the local error
        estimates less than ``atol + rtol * abs(y)``. Here, `rtol` controls a
        relative accuracy (number of correct digits). But if a component of `y`
        is approximately below `atol`, the error only needs to fall within
        the same `atol` threshold, and the number of correct digits is not
        guaranteed. If components of y have different scales, it might be
        beneficial to set different `atol` values for different components by
        passing tensor with shape (neqs,) for `atol`. Default values are
        1e-3 for `rtol` and 1e-6 for `atol`.
    Attributes
    ----------
    n : int
        Number of batchs.
    status : string
        Current status of the solver: 'running', 'finished' or 'failed'.
    t_bound : float
        Boundary time.
    direction : float
        Integration direction: +1 or -1.
    t : float
        Current time.
    y : tensor
        Current state.
    t_old : float
        Previous time. None if no steps were made yet.
    step_size : float
        Size of the last successful step. None if no steps were made yet.
    nfev : int
        Number evaluations of the system's right-hand side.
    References
    ----------
    .. [1] P. Bogacki, L.F. Shampine, "A 3(2) Pair of Runge-Kutta Formulas",
    Appl. Math. Lett. Vol. 2, No. 4. pp. 321-325, 1989.
    """
    order = 3
    error_estimator_order = 2
    n_stages = 3
    C = torch.tensor([0., 1./2, 3./4])
    A = torch.tensor([[0., 0., 0.], [1./2, 0., 0.], [0., 3./4, 0.]])
    B = torch.tensor([2./9, 1./3, 4./9])
    E = torch.tensor([5./72, -1./12, -1./9, 1./8])
    P = torch.tensor([[1., -4./3, 5./9], [0., 1., -2./3],
                      [0., 4./3, -8./9], [0., -1., 1.]])


class SSPRK23(AdaptativeRungeKutta):
    """Explicit Strong Stability Preserving Runge-Kutta (SSPRK) method of order 3(2).
    This uses the optimal SSPRK(4,3) method with embedded pair [1/3,1/3,1/3,0]
    of [1]_.The error is controlled assuming accuracy of the second-order method,
    but steps are taken using the third-order accurate formula. A second-order
    SSP dense output is used for the local interpolation according to [2]_.
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
    y0 : array_like, shape (B, neqs, n_points)
        Initial state.
    t_bound : float
        Boundary time - the integration won't continue beyond it. It also
        determines the direction of the integration.
    first_step : float or None, optional
        Initial step size. Default is ``None`` which means that the algorithm
        should choose.
    max_step : float, optional
        Maximum allowed step size. Default is np.inf, i.e., the step size is not
        bounded and determined solely by the solver.
    rtol, atol : float and array_like, optional
        Relative and absolute tolerances. The solver keeps the local error
        estimates less than ``atol + rtol * abs(y)``. Here, `rtol` controls a
        relative accuracy (number of correct digits). But if a component of `y`
        is approximately below `atol`, the error only needs to fall within
        the same `atol` threshold, and the number of correct digits is not
        guaranteed. If components of y have different scales, it might be
        beneficial to set different `atol` values for different components by
        passing tensor with shape (neqs,) for `atol`. Default values are
        1e-3 for `rtol` and 1e-6 for `atol`.
    Attributes
    ----------
    n : int
        Number of batchs.
    status : string
        Current status of the solver: 'running', 'finished' or 'failed'.
    t_bound : float
        Boundary time.
    direction : float
        Integration direction: +1 or -1.
    t : float
        Current time.
    y : tensor
        Current state.
    t_old : float
        Previous time. None if no steps were made yet.
    step_size : float
        Size of the last successful step. None if no steps were made yet.
    nfev : int
        Number evaluations of the system's right-hand side.
    References
    ----------
    .. [1] S. Conde, I. Fekete and J.N. Shadid, "Embedded error estimation and
    adaptive step-size control for optimal explicit strong stability
    preserving Runge--Kutta methods", Preprint:arXiv:1806.08693, 2018.
    .. [2] D.I. Ketcheson, L. Lóczi, A. Jangabylova, A. Kusmanov, "Dense output
    for strong stability preserving Runge-Kutta methods", arXiv:1605.02429v2
    [math.NA] , 2016.
    """
    order = 3
    error_estimator_order = 2
    n_stages = 4
    C = torch.tensor([0., 1./2, 1., 1./2])
    A = torch.tensor([[0., 0., 0., 0.], [1./2, 0., 0., 0.],
                      [1./2, 1./2, 0., 0.], [1./6, 1./6, 1./6, 0.]])
    B = torch.tensor([1./6, 1./6, 1./6, 1./2])
    # E = torch.tensor([1./12, 1./12, 1./12, -1./4, 0.])
    E = torch.tensor([1./6, 1./6, 1./6, -1./2, 0.])
    P = torch.tensor([[1., -5./6], [0., 1./6], [0., 1./6], [0., 1./2], [0., 0.]])


class RK5SSP3(AdaptativeRungeKutta):
    """Explicit Runge-Kutta method of order 5(3).
    This uses the six stages fifth-order RK method with an embedded optimal three stage
    third-order SSP method, denoted RK6(5) / SSP3(3), as detailed in [1]_, Table 4.12.
    The error is controlled assuming accuracy of the fourth-order method accuracy, but
    steps are taken using the fifth-order accurate formula (local extrapolation is done).
    A quartic interpolation polynomial is used for the dense output [2]_.
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
    y0 : tensor
        Initial state.
    t_bound : float
        Boundary time - the integration won't continue beyond it. It also
        determines the direction of the integration.
    first_step : float or None, optional
        Initial step size. Default is ``None`` which means that the algorithm
        should choose.
    max_step : float, optional
        Maximum allowed step size. Default is np.inf, i.e., the step size is not
        bounded and determined solely by the solver.
    rtol, atol : float and array_like, optional
        Relative and absolute tolerances. The solver keeps the local error
        estimates less than ``atol + rtol * abs(y)``. Here `rtol` controls a
        relative accuracy (number of correct digits). But if a component of `y`
        is approximately below `atol`, the error only needs to fall within
        the same `atol` threshold, and the number of correct digits is not
        guaranteed. If components of y have different scales, it might be
        beneficial to set different `atol` values for different components by
        passing array_like with shape (neqs,) for `atol`. Default values are
        1e-3 for `rtol` and 1e-6 for `atol`.
    vectorized : bool, optional
        Whether `fun` is implemented in a vectorized fashion. Default is False.
    Attributes
    ----------
    n : int
        Number of batchs.
    status : string
        Current status of the solver: 'running', 'finished' or 'failed'.
    t_bound : float
        Boundary time.
    direction : float
        Integration direction: +1 or -1.
    t : float
        Current time.
    y : tensor
        Current state.
    t_old : float
        Previous time. None if no steps were made yet.
    step_size : float
        Size of the last successful step. None if no steps were made yet.
    nfev : int
        Number evaluations of the system's right-hand side.
    References
    ----------
    .. [1] C.B. Macdonald, "Constructing high-order Runge-Kutta methods with
    embedded strong-stability-preserving pairs", MasterThesis, 2003.
    .. [2] L. W. Shampine, "Some Practical Runge-Kutta Formulas", Mathematics
    of Computation, Vol. 46, No. 173, pp. 135-150, 1986.
    """
    order = 5
    error_estimator_order = 3
    n_stages = 6
    C = torch.tensor([0., 1., 1./2, 1./5, 2./3, 1.])
    A = torch.tensor([
        [0., 0., 0., 0., 0.],
        [1., 0., 0., 0., 0.],
        [1./4, 1./4, 0., 0., 0.],
        [2046./15625, -454./15625, 1533./15625, 0., 0.],
        [-739./5625, 511./5625, -566./16875, 20./27, 0.],
        [11822./21875, -6928./21875, -4269./21875, -4./7, 54./35]])
    B = torch.tensor([1./24, 0., 0., 125./336, 27./56, 5./48])
    E = torch.tensor(
        [1./8, 1./6., 2./3, -125./336, -27./56, -5./48, 0.])
    P = torch.tensor([
        [1., -15./4, 14./3, -15./8],
        [0., 0., 0., 0.],
        [0., 0., 0., 0.],
        [0., 125./28, -625./84, 375./112],
        [0., -27./28, 27./7, -135./56],
        [0., 1./4, -13./12, 15./16],
        [0., 0., 0., 0.]])
    
    

class RK45_DP(AdaptativeRungeKutta):
    """Explicit Runge-Kutta method of order 5(4).
    This uses the Dormand-Prince pair of formulas [1]_. The error is controlled
    assuming accuracy of the fourth-order method accuracy, but steps are taken
    using the fifth-order accurate formula (local extrapolation is done). A quartic
    interpolation polynomial is used for the dense output [2]_.
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
    y0 : tensor
        Initial state.
    t_bound : float
        Boundary time - the integration won't continue beyond it. It also
        determines the direction of the integration.
    first_step : float or None, optional
        Initial step size. Default is ``None`` which means that the algorithm
        should choose.
    max_step : float, optional
        Maximum allowed step size. Default is np.inf, i.e., the step size is not
        bounded and determined solely by the solver.
    rtol, atol : float and array_like, optional
        Relative and absolute tolerances. The solver keeps the local error
        estimates less than ``atol + rtol * abs(y)``. Here `rtol` controls a
        relative accuracy (number of correct digits). But if a component of `y`
        is approximately below `atol`, the error only needs to fall within
        the same `atol` threshold, and the number of correct digits is not
        guaranteed. If components of y have different scales, it might be
        beneficial to set different `atol` values for different components by
        passing array_like with shape (neqs,) for `atol`. Default values are
        1e-3 for `rtol` and 1e-6 for `atol`.
    vectorized : bool, optional
        Whether `fun` is implemented in a vectorized fashion. Default is False.
    Attributes
    ----------
    n : int
        Number of batchs.
    status : string
        Current status of the solver: 'running', 'finished' or 'failed'.
    t_bound : float
        Boundary time.
    direction : float
        Integration direction: +1 or -1.
    t : float
        Current time.
    y : tensor
        Current state.
    t_old : float
        Previous time. None if no steps were made yet.
    step_size : float
        Size of the last successful step. None if no steps were made yet.
    nfev : int
        Number evaluations of the system's right-hand side.
    References
    ----------
    .. [1] J. R. Dormand, P. J. Prince, "A family of embedded Runge-Kutta
    formulae", Journal of Computational and Applied Mathematics, Vol. 6,
    No. 1, pp. 19-26, 1980.
    .. [2] L. W. Shampine, "Some Practical Runge-Kutta Formulas", Mathematics
    of Computation, Vol. 46, No. 173, pp. 135-150, 1986.
    """
    order = 5
    error_estimator_order = 4
    n_stages = 6
    C = torch.tensor([0., 1./5, 3./10, 4./5, 8./9, 1.])
    A = torch.tensor([
        [0., 0., 0., 0., 0.], [1./5, 0., 0., 0., 0.],
        [3./40, 9./40, 0., 0., 0.], [44./45, -56./15, 32./9, 0., 0.],
        [19372./6561, -25360./2187, 64448./6561, -212./729, 0.],
        [9017./3168, -355./33, 46732./5247, 49./176, -5103./18656]])
    B = torch.tensor([35./384, 0., 500./1113, 125./192, -2187./6784, 11./84])
    E = torch.tensor(
        [-71./57600, 0., 71./16695, -71./1920, 17253./339200, -22./525, 1./40])
    # Corresponds to the optimum value of c_6 from [2]_.
    P = torch.tensor([
        [1., -8048581381./2820520608, 8663915743./2820520608,
         -12715105075./11282082432],
        [0., 0., 0., 0.],
        [0., 131558114200./32700410799, -68118460800./10900136933,
         87487479700./32700410799],
        [0., -1754552775./470086768, 14199869525./1410260304,
         -10690763975./1880347072],
        [0., 127303824393./49829197408, -318862633887./49829197408,
         701980252875./199316789632],
        [0., -282668133./205662961, 2019193451./616988883, -1453857185./822651844],
        [0., 40617522./29380423, -110615467./29380423, 69997945./29380423]])


class RkDenseOutput(DenseOutput):

    def __init__(self, t_min, t_max, y_old, Q):
        super().__init__(t_min, t_max)
        self.h = t_max - t_min
        self.Q = Q
        self.order = Q.shape[-1]
        self.y_old = y_old

    def _call_impl(self, t):
        x = (t - self.t_min) / self.h
        if torch.numel(x) == 1:
            p = torch.tile(x, (self.order,1))
            p = torch.cumprod(p, dim=0)
        else:
            p = x.repeat_interleave(self.order).reshape(-1, self.order).T
            p = torch.cumprod(p, dim=0)
        y = self.h * self.Q @ p
        y += self.y_old[..., None]

        return [y[...,i] for i in range(y.shape[-1])]


def _rk_step(fun, t, y, h, A, B, C, K):
    """Perform a single Runge-Kutta step.
    This function computes a prediction of an explicit Runge-Kutta method.
    Notation for Butcher tableau is as in [1]_.
    Parameters
    ----------
    fun : callable
        Right-hand side of the system.
    t : float
        Current time.
    y : tensor, shape (B, neqs, n_points)
        Current state.
    h : float
        Step to use.
    A : tensor, shape (n_stages, n_stages)
        Coefficients for combining previous RK stages to compute the next
        stage. For explicit methods the coefficients at and above the main
        diagonal are zeros.
    B : tensor, shape (n_stages)
        Coefficients for combining RK stages for computing the final
        prediction.
    C : tensor, shape (n_stages)
        Coefficients for incrementing time for consecutive RK stages.
        The value for the first stage is always zero.
    K : tensor, shape (B, neqs, n_points, n_stages)
        Storage array for putting RK stages here. The last stage is a linear
        combination of the previous stages with coefficients of B.
    Returns
    -------
    y_new : tensor, shape (B, neqs, n_points)
        Solution at t + h computed with a higher accuracy.
    References
    ----------
    .. [1] E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential
           Equations I: Nonstiff Problems", Sec. II.4.
    """
    K[...,0] = fun(t, y)
    for s, (a, c) in enumerate(zip(A[1:], C[1:]), start=1):
        dy = h * K[...,:s] @ a[:s]
        K[...,s] = fun(t + c * h, y + dy)

    y_new = y + h * K @ B

    return y_new


class RungeKutta(OdeSolver):
    """Base class for explicit adaptative Runge-Kutta methods.
    """
    C: torch.Tensor = NotImplemented
    A: torch.Tensor = NotImplemented
    B: torch.Tensor = NotImplemented
    order: int = NotImplemented
    error_estimator_order: int = NotImplemented
    n_stages: int = NotImplemented

    def __init__(self, fun, t0, y0, t_bound, rtol=1e-3, atol=1e-6, max_step=FMAX,
                 first_step=None, **extraneous):
        warn_extraneous(extraneous)
        super().__init__(fun, t0, y0, t_bound)
        self.min_step = 10 * EPS
        self.max_step = validate_max_step(max_step)
        self.rtol, self.atol = validate_tol(rtol, atol, self.n)
        self.f = self.fun(self.t, self.y)

        if first_step is None:
            self.h = select_initial_step(
                self.fun, self.t, self.y, self.f, 1.0, self.error_estimator_order,
                self.rtol, self.atol)
        else:
            self.h = validate_first_step(first_step, t0, t_bound)
        if self.h > self.max_step:
            self.h = self.max_step
        elif self.h < self.min_step:
            self.h = self.min_step
        self.K = torch.empty(y0.shape + (self.n_stages,), dtype=y0.dtype)

    def _step_impl(self):

        t_new = self.t + self.h
        if t_new > self.t_bound:
            t_new = self.t_bound
            h = t_new - self.t
        else:
            h = self.h

        y_new = _rk_step(self.fun, self.t, self.y, h, self.A,
                         self.B, self.C, self.K)
        self.t = t_new
        self.y = y_new

        return True, None

    # hacky way of dynamically moving the tensors to the correct device
    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs) 
        self.A = self.A.to(*args, **kwargs)
        self.B = self.B.to(*args, **kwargs)
        self.C = self.C.to(*args, **kwargs)
        self.K = self.K.to(*args, **kwargs)
        return self

    #TODO: Implement local interpolants (E matrix) for each non adaptative
    # method depending on wheter they are SSP or not!
    # def _dense_output_impl(self):
    #     Q = self.K @ self.P
    #     return RkDenseOutput(self.t_old, self.t, self.y_old, Q)


class RK2(RungeKutta):
    """Explicit Runge-Kutta method of order 2."""
    order = 2
    n_stages = 2
    C = torch.tensor([0., 1./2])
    A = torch.tensor([[0., 0.], [1./2, 0.]])
    B = torch.tensor([0., 1.])


class HEUN2(RungeKutta):
    """Explicit Heun method of order 2."""
    order = 2
    n_stages = 2
    C = torch.tensor([0., 1.])
    A = torch.tensor([[0., 0.], [1., 0.]])
    B = torch.tensor([1./2, 1./2])


class RK3(RungeKutta):
    """Explicit Runge-Kutta method or order 3."""
    order = 3
    n_stages = 3
    C = torch.tensor([0., 1./2, 1.])
    A = torch.tensor([[0., 0., 0.], [1./2, 0., 0.], [-1., 2., 0.]])
    B = torch.tensor([1./6, 2./3, 1./6])


class HEUN3(RungeKutta):
    """Explicit Heun method of order 3."""
    order = 3
    n_stages = 3
    C = torch.tensor([0., 1./3, 2./3])
    A = torch.tensor([[0., 0., 0.], [1./3, 0., 0.], [0., 2./3, 0.]])
    B = torch.tensor([1./4, 0., 3./4])


class SSPRK3(RungeKutta):
    """Strong Stability Preserving Explicit Runge-Kutta method or order 3:
        https://doi.org/10.1016/0021-9991(88)90177-5
    """
    order = 3
    n_stages = 3
    C = torch.tensor([0., 1., 1./2])
    A = torch.tensor([[0., 0., 0.], [1., 0., 0.], [1./4, 1./4, 0.]])
    B = torch.tensor([1./6, 1./6, 2./3])


class RALSTON3(RungeKutta):
    """Ralston's method of order 3 used in the embedded Bogacki–Shampine method
    RK3(2).
    """
    order = 3
    n_stages = 3
    C = torch.tensor([0., 1./2, 3./4])
    A = torch.tensor([[0., 0., 0.], [1./2, 0., 0.], [0., 3./4, 0.]])
    B = torch.tensor([2./9, 1./3, 4./9])


class RK4(RungeKutta):
    """Explicit Runge-Kutta method or order 4."""
    order = 4
    n_stages = 4
    C = torch.tensor([0., 1./2, 1./2, 1.])
    A = torch.tensor([[0., 0., 0., 0.], [1./2, 0., 0., 0.],
                      [0., 1./2, 0., 0.], [0., 0., 1., 0.]])
    B = torch.tensor([1./6, 1./3, 1./3, 1./6])


class RK4_38(RungeKutta):
    """A less notorious explicit Runge-Kutta method  of order 4, introduced in
    the same paper as RK4."""
    order = 4
    n_stages = 4
    C = torch.tensor([0., 1./3, 2./3, 1.])
    A = torch.tensor([[0., 0., 0., 0.], [1./3, 0., 0., 0.],
                      [-1./3., 1., 0., 0.], [1., -1., 1., 0.]])
    B = torch.tensor([1./8, 3./8, 3./8, 1./8])


class RK5_3(RungeKutta):
    """The non adptative version of RK5SPP3."""
    order = 5
    n_stages = 6
    C = torch.tensor([0., 1., 1./2, 1./5, 2./3, 1.])
    A = torch.tensor([
        [0., 0., 0., 0., 0.],
        [1., 0., 0., 0., 0.],
        [1./4, 1./4, 0., 0., 0.],
        [2046./15625, -454./15625, 1533./15625, 0., 0.],
        [-739./5625, 511./5625, -566./16875, 20./27, 0.],
        [11822./21875, -6928./21875, -4269./21875, -4./7, 54./35]])
    B = torch.tensor([1./24, 0., 0., 125./336, 27./56, 5./48])


class RALSTON4(RungeKutta):
    """Ralston's method of order 4."""
    order = 4
    n_stages = 4
    C = torch.tensor([0., 0.4, 0.45573725, 1.])
    A = torch.tensor([[0., 0., 0., 0.],
                      [0.4, 0., 0., 0.],
                      [.29697761, .15875964, 0., 0.],
                      [.21810040, -3.05096516, 3.83286476, 0.]])
    B = torch.tensor([.17476028, -.55148066, 1.20553560, .17118478])


class EULER(OdeSolver):
    """Basic Euler scheme."""
    order = 1
    n_stages = 1

    def __init__(self, fun, t0, y0, t_bound, rtol=1e-3, atol=1e-6, max_step=FMAX,
                 first_step=None, **extraneous):
        warn_extraneous(extraneous)
        super().__init__(fun, t0, y0, t_bound)
        self.min_step = 10 * EPS
        self.max_step = validate_max_step(max_step)
        self.rtol, self.atol = validate_tol(rtol, atol, self.n)
        self.f = self.fun(self.t, self.y)

        if first_step is None:
            self.h = select_initial_step(
                self.fun, self.t, self.y, self.f, 1.0, self.error_estimator_order,
                self.rtol, self.atol)
        else:
            self.h = validate_first_step(first_step, t0, t_bound)
        if self.h > self.max_step:
            self.h = self.max_step
        elif self.h < self.min_step:
            self.h = self.min_step

    def _step_impl(self):

        t_new = self.t + self.h
        if t_new > self.t_bound:
            t_new = self.t_bound
            h = t_new - self.t
        else:
            h = self.h

        y_new = self.y + h * self.fun(self.t, self.y)
        self.t = t_new
        self.y = y_new

        return True, None
