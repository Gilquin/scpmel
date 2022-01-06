"""
@author: Laurent Gilquin

This file is specific to each model instance as it requires importing the
corresponding model class, its exogeneous submodule(s) and instantiating both.
"""

# global imports
import sys
import h5py
# relative imports
from kdv import KDV
from exogenous import EnhancedSmagorinsky


def _unpack_grp(group):
    """
    Unpack datasets of an hdf5 group into a dictionnary.
    """
    dict_out = dict({ele[0]: ele[1][()] for ele in group.items()})
    return dict_out


def load_model(path, cutoff, dt, verbose=True):
    """
    Utility function to instantiate a Pytorch model.

    Parameters
    ----------
    path : str
        The path where to find the data file.
    cutoff : int
        The cutoff number used for instantiating the model.
    dt : float
        The length scale of the time scheme.

    Returns
    -------
    An instance of the model (torch.nn.Module)
    """

    # get configurations
    with h5py.File(path, mode="r", libver="latest") as les_file:
        space_config = _unpack_grp(les_file["config/space"])
        time_config = _unpack_grp(les_file["config/time"])
        kdv_params = _unpack_grp(les_file["config/params"])
        options = _unpack_grp(les_file["config/options"])
        cutoff_config = dict(**les_file[cutoff].attrs)
    # zero out the artificial dissipation
    kdv_params['nu'] = 0.0

    # instantiate model
    scale = space_config['x_end'] - space_config['x_start']
    dx = cutoff_config['cutoff'] / 2
    if verbose:
        print("LES cutoff length equals: {}".format(cutoff_config['cutoff']))
        print("LES coarse grid step equals: {}".format(dx))
        print("LES coarse grid number of points equals: {}".format(scale//dx))

    # activate WENO-DS scheme
    WENO_DS = True
    model = KDV((scale//dx,), (scale,), **kdv_params, DS=WENO_DS)

    # instantiate subgrid model
    TauNet = EnhancedSmagorinsky(cutoff_config['cutoff'], scale, acc=4, bounds=[0.0, 0.2])
    model.make_exogenous({'tau': TauNet})

    # set time configuration
    time_horizon = float(time_config["t_step"])
    time_scheme = "SSPRK3"
    if verbose:
        print("time scheme: {}".format(time_scheme))
        print("time horizon: {}".format(time_horizon))
        print("dt equals: {}".format(dt))
    model.set_time(dt, [0.0, time_horizon], [time_horizon], method=time_scheme,
                   dense_output=False, **options)

    return model


if __name__ == '__main__':
    
    import torch
    from scpmel.utils import tools
    torch.set_default_dtype(torch.float64)

    path = "./data/KDV_LES_800.0.hdf5"
    cutoff = "cutoff1"
    dt = 80.0
    try:
        kdv = load_model(path, cutoff, dt)
        kdv.to("cuda:0")
        print(tools.summary(kdv, (1, int(kdv.shape[-1])), 20, device="cuda:0", dtypes=[torch.float64]))
    except:
        print("Unexpected error:", sys.exc_info()[0])
        raise
