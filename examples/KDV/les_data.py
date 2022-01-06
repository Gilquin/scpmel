"""
@author: Laurent Gilquin

Creates the LES dataset based on the following arguments:
    * path: the absolute path to the hdf5 file containing the DNS dataset,
    * time-step: the time stepping used to train the model with the pair (input, target)
    defined as: ( u(t), u(t+time-step)). It must be a multiple of the DNS dataset
    't_step' attribute.
    * skip: can be used to skip chuncks of the DNS dataset if the latter is too
    large. For instance, 'skip=2' implies that only one every two data is kept 
    (e.g. half the DNS dataset is dropped). Defaults to 1 (nothing dropped).
"""

# global imports
import argparse
import os
import h5py
import numpy as np
import torch
# relative imports
from scpmel.les import get_filter
from scpmel.training import splitter

# set torch dtype to double
torch.set_default_dtype(torch.float64)

# parser setting with default values and actions
parser = argparse.ArgumentParser(description='KDV - creation of LES datasets.')
parser.add_argument('path', metavar='DIR', help='absolute path to the hdf5 '
                    'DNS dataset.')
parser.add_argument('--time-step', type=float, metavar='N',
                    help='The time step used to form the LES database.')
parser.add_argument('--ratios', type=str, metavar='N',
                    help='The subsampling ratios of the DNS grid as a comma '
                    'separated string.')
parser.add_argument('--skip', default=1, type=int, metavar='N',
                    help='The time sub-sampling ratio to reduce the database '
                    '(default 50).')

def main():

    # get the command line options
    args = parser.parse_args()

    # create HD5 file for the given time step
    filename = args.path.split("/")[-1]
    filename = filename.replace("DNS","LES_" + str(args.time_step))
    path = os.path.join(os.path.dirname(args.path), filename)
    les_file = h5py.File(path, "w", libver="latest")

    # recover the DNS config, the trajectory keys
    with h5py.File(args.path, "r", libver="latest") as dns_file:
        # get the simulation keys
        sim_keys = [ ele for ele in dns_file.keys() if ele!="config"]
        # restrict the time keys based on the prescribed time step
        time_jump = max(1, int(args.time_step/dns_file["config/time/t_step"][()]))
        # get the spatial config
        x_start = dns_file["config/space/x_start"][()]
        x_end = dns_file["config/space/x_end"][()]
        dx = dns_file["config/space/dx"][()]
    # set the dns_grid
    scale = x_end - x_start
    dns_grid = np.linspace(x_start, x_end, int(scale/dx), endpoint=False)

    # store config with new time step
    with h5py.File(args.path, "r", libver="latest") as dns_file:
        dns_file.copy("config", les_file)
    les_file["config/time/t_step"][()] = args.time_step
    del les_file["config/space/dx"]

    # store the data
    data_grp = les_file.create_group("data")
    with h5py.File(args.path, "r", libver="latest") as dns_file:
        for simu in sim_keys:
            # get the time keys
            time_keys = list(dns_file[simu].keys())
            for key_idx in range(len(time_keys))[::args.skip]:
                if key_idx<(len(time_keys)-time_jump):
                    key = time_keys[key_idx]
                    key_str = simu+'_'+key
                    next_key = time_keys[key_idx+time_jump]
                    next_key_str = simu+'_'+next_key
                    if key_str not in list(data_grp.keys()): # avoid duplicate error
                        data_grp.create_dataset(
                            name=key_str, data=dns_file[simu][key][:],
                            compression="gzip")
                    if next_key_str not in list(data_grp.keys()): # avoid duplicate error
                        data_grp.create_dataset(
                            name=next_key_str, data=dns_file[simu][next_key][:],
                            compression="gzip")

    # define subsampling ratios and cutoffs
    ratios = np.fromstring(args.ratios, dtype=int, sep=",")
    cutoffs = dx * ratios

    # build and fill cutoff groups
    try:
        with h5py.File(args.path, "r", libver="latest") as dns_file:
            # iterate over cutoff lengths
            for idx, cutoff in enumerate(cutoffs):
                # create cutoff group
                cutoff_grp = les_file.create_group("cutoff"+str(idx))
                cutoff_grp.attrs["cutoff"] = cutoff
                cutoff_grp.attrs["ratio"] = ratios[idx]
                # store corresponding Gaussian filter
                _, g_filter = get_filter(dns_grid, cutoff, "Box", tol=1e-8)
                cutoff_grp.create_dataset(
                    name="filter", data=g_filter, compression="gzip")
                # iterate over set of simulations
                train_keys = []
                valid_keys = []
                for simu in sim_keys:
                    # get the training and validation keys
                    train_idx, valid_idx = splitter(time_keys[::args.skip], 0.1)
                    train_idx = [item * args.skip for item in train_idx]
                    valid_idx = [item * args.skip for item in valid_idx]
                    train_keys += [
                        (simu+'_'+time_keys[idx], simu+'_'+time_keys[idx+time_jump]) for
                        idx in train_idx if idx<(len(time_keys)-time_jump)]
                    valid_keys += [
                        (simu+'_'+time_keys[idx], simu+'_'+time_keys[idx+time_jump]) for
                        idx in valid_idx if idx<(len(time_keys)-time_jump)]
                # store the full set of training and validation keys
                cutoff_grp.create_dataset(
                    name="training_keys", data=train_keys, compression="gzip")
                cutoff_grp.create_dataset(
                    name="validation_keys", data=valid_keys, compression="gzip")
        les_file.close()
    except:
        les_file.close()


if __name__ == '__main__':
    main()
