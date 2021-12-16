"""
Created on Tue Aug 31 12:07:48 2021

Tests of solve_ip and IVP calls with toy models (borrowed from scipy and others).

@author: Laurent Gilquin
"""

import torch
from .ivp import IVP, solve_ivp, METHODS
#%%
# Basic exponential decay 
def exponential_decay(t, y): return -0.5 * y

# automatically chosen time points with default method (RK45)
t_span = [0, 10]
y0 = torch.tensor([[[2.]], [[4.]], [[8.]]], dtype=torch.get_default_dtype())
sol = solve_ivp(exponential_decay, t_span, y0)
print(sol.t)
print(sol.y)

# varying initial condition shape
t_span = [0, 10]
y0 = torch.tensor([[[2.]], [[4.]], [[8.]]], dtype=torch.get_default_dtype())
sol = solve_ivp(exponential_decay, t_span, y0)
print(sol.t)
print(sol.y)

# specified time points with default method (RK45)
t_span = [0, 10]
t_eval = [0, 1, 2, 4, 10]
y0 = torch.tensor([2., 4., 8.], dtype=torch.get_default_dtype())
sol = solve_ivp(exponential_decay, t_span, y0, t_eval=t_eval, dense_output=True)
print(sol.t)
print(sol.y)
#%%
# test all method with automatic time points selection
for method in METHODS.keys():
    sol = solve_ivp(exponential_decay, t_span, y0, method=method)
    print(method, sol.y[-1])
    if not sol.success:
        print("{} ended with message: {}".format(method, sol.message))
#%%
# test the IVP interface
ivp_c = IVP()

# no specified times -> dense output argument doesnt matter 
t_span = [0, 10]
t_eval = None
y0 = torch.tensor([[[2.]], [[4.]], [[8.]]], dtype=torch.get_default_dtype())
ivp_c.set_time(None, t_span, t_eval, method="RK45_DP", dense_output=True)
sol = ivp_c.forecast(exponential_decay, y0, full=True) # class
sol = ivp_c.forecast(exponential_decay, y0, full=False) # list of tensors

# specified times
t_eval = [0, 1, 2, 4, 10]
# dense output -> only the t_eval states are returned
ivp_c.set_time(None, t_span, t_eval, method="RK45_DP", dense_output=True)
sol = ivp_c.forecast(exponential_decay, y0, full=True)
# no dense output -> only the final state is returned
ivp_c.set_time(None, t_span, t_eval, method="RK45_DP", dense_output=False)
sol = ivp_c.forecast(exponential_decay, y0, full=True)

# test non adaptative method with user defined time step
t_span = [0., 10.]
t_eval = [0.07, 1.75, 3.5, 5.25, 7.0, 8.75, 9.94]
# return nothing since dt is selected by the solver and t is never in t_eval
ivp_c.set_time(None, t_span, t_eval, method="SSPRK3", dense_output=False)
sol = ivp_c.forecast(exponential_decay, y0, full=True)
# return at specified t_eval times
dt = 0.07
ivp_c.set_time(dt, t_span, t_eval, method="SSPRK3", dense_output=False)
sol = ivp_c.forecast(exponential_decay, y0, full=True)
# throw NotImplementedError
ivp_c.set_time(dt, t_span, t_eval, method="SSPRK3", dense_output=True)
sol = ivp_c.forecast(exponential_decay, y0, full=True)
