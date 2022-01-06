"""
@author: Laurent Gilquin

Creates the DNS dataset based on the following arguments:
    * path: the absolute path where the hdf5 file containing the DNS dataset will be stored.
    * nsim: the number of simulations to perform.
"""

# global imports
import argparse
import h5py
import numpy as np
import os
import torch
# relative imports
from kdv import KDV
from scpmel.utils import tools
from scpmel.sampling import custom_samples

# set torch dtype to double
torch.set_default_dtype(torch.float64)

parser = argparse.ArgumentParser(description='KDV - creation of DNS dataset.')
parser.add_argument('path', metavar='DIR', help='absolute path to the directory where the'
                    'dataset will be stored.')
parser.add_argument('--nsim', default=10, type=int, metavar='N',
                    help='The number of simulations to perform.')

def main():

    # get the command line options
    args = parser.parse_args()        

    # spatial steup
    x_start = -2**15
    x_end = 2**15
    x_scale = x_end - x_start
    npts = 1024
    dx = x_scale / npts
    dns_grid = np.linspace(x_start, x_end, npts, endpoint=False)

    # time setup
    t_burnin = 3600 * 24
    t_end = t_burnin * 2
    t_step = 100
    t_scheme = "RK5SSP3"
    t_eval = torch.arange(t_burnin, t_end+0.1, t_step).tolist()

    # kdv parameters
    h1, h2 = (200, 1200)
    g = 0.03
    c0 = np.sqrt(h1*h2/(h1+h2)*g)
    A = 3/2*(h1-h2)/(h1*h2)*c0
    B = 1/6*c0*h1*h2
    nu = 0.25
    w_pow = 2
    # KDV model instantiation
    dns_kdv = KDV((x_scale//dx,), (x_scale,), c0=c0, A=A, B=B, nu=nu, w_pow=w_pow)
    dns_kdv.make_exogenous({'tau': lambda x: 0.0}) # no closure

    # tolerance options for adaptative time stepping
    options = {'rtol': 1e-4, 'atol': 1e-6}

    # fix seeds for reproducibility
    seeds = torch.randint(np.iinfo(np.int32).max, size=(args.nsim,)).numpy()

    # store whole configuration
    config = {
        "time": {
            "t_burnin": t_burnin,
            "t_end": t_end,
            "t_step": t_step,
            "t_eval": t_eval,
            "t_scheme": t_scheme
            },
        "params": {
            "c0": c0,
            "A": A,
            "B": B,
            "nu": nu,
            "w_pow": w_pow,
            },
        "space": {
            "x_start": x_start,
            "x_end": x_end,
            "dx": dx
            },
        "options": options,
        "seeds": dict({*enumerate(seeds)})
        }

    # instatiate DNS file
    path = os.path.join(args.path, "KDV_DNS.hdf5")
    # save config
    with h5py.File(path, "w", libver="latest") as dns_file:
        config_grp = dns_file.create_group("config")
        for name, value in config.items():
            subgrp = config_grp.create_group(name)
            for key, val in value.items():
                subgrp.create_dataset(name=str(key), data=val)

    # loop over scenarii
    str_format = tools.format_time([t_end], t_step)
    for idx, seed in enumerate(seeds):

        print("runing simulation {}".format(idx))

        # initial condition
        u0 = custom_samples(dns_grid, amp=100, N=5, lset=[10,15,20,25], seed=seed)
        u0 = torch.tensor(u0, dtype=torch.get_default_dtype()).reshape(1,1,-1)

        # burnin phase (no dissipation)
        dns_kdv.nu = 0
        t_span = [0, t_burnin]
        t_eval = torch.arange(t_span[0], t_span[1]+0.1, t_step).tolist()
        dns_kdv.set_time(None, t_span, t_eval, method=t_scheme, dense_output=True, **options)
        traj_dns = dns_kdv.predict(u0) # autograd disabled
        torch.save(traj_dns, os.path.join(args.path, "sim"+str(idx)+"_day1.pt"))

        # simulation phase (with dissipation)
        u0 = traj_dns.y[-1]
        dns_kdv.nu = nu
        t_span = [t_burnin, t_end]
        t_eval = torch.arange(t_span[0], t_span[1]+0.1, t_step).tolist()
        dns_kdv.set_time(None, t_span, t_eval, method=t_scheme, dense_output=True, **options)
        traj_dns = dns_kdv.predict(u0) # autograd disabled
        torch.save(traj_dns, os.path.join(args.path, "sim"+str(idx)+"_day2.pt"))

        # store simulations
        with h5py.File(path, "a", libver="latest") as dns_file:
            # store simulations
            data_grp = dns_file.create_group("sim" + str(idx))
            # index simulations by times
            for idx, key in enumerate(t_eval):
                data_grp.create_dataset(
                    name=str_format % key, data=traj_dns.y[idx][0,0,:], compression="gzip")


if __name__ == '__main__':
    main()
