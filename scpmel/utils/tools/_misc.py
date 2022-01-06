# global imports
from collections import OrderedDict
from decimal import Decimal
import errno
import inspect
import os
import torchinfo

__all__ = ["mkdir_p", "format_time", "format_state_dict", "get_args", "summary"]

def mkdir_p(path):
    '''make dir if not exist'''
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def format_time(times, dt):
    """
    Get the best time format with decimal accuracy provided by dt. See:
        * discusion on decimal accuracy:
            (https://stackoverflow.com/questions/6189956/easy-way-of-finding-decimal-places)
        * discussion on formatting float:
            (https://stackoverflow.com/questions/1424638/pad-python-floats)
    """
    int_acc = max([len(str(int(time))) for time in times])
    decim_acc = abs(Decimal(str(dt)).as_tuple().exponent)
    tot_digits = int_acc + decim_acc
    if decim_acc > 0:
        tot_digits += 1 # 1 for the dot
        t_string = '%0{}.{}f'.format(tot_digits, decim_acc)
    else:
        t_string = '%0{}d'.format(tot_digits)

    return t_string


def format_state_dict(state_dict, load_into_parallel):
    """
    Utility function to correctly format the state dictionnary of a torch.nn.Module
    whether it is loaded into a DataParallel instance or not. See the following
    discussion for more informations:
        https://github.com/bearpaw/pytorch-classification/issues/27

    Parameters
    ----------
    state_dict  : dict
        The state dictionnary to be formatted.    
    load_into_parallel : bool
        Whether the state dictionnary is loaded into a DataParallel instance or
        not.

    Returns
    -------
    The correctly formatted state dictionnary.
    """

    if not load_into_parallel:
        # remove the module. key at the beginning of each feature if needed
        correct_dict = OrderedDict(**{
            key.replace("module.", "") if 'module' in key else key: value
            for key, value in state_dict.items()})
    else:
        correct_dict = OrderedDict(**{
            "module."+key if 'module' not in key else key: value
            for key, value in state_dict.items()})

    return correct_dict



def get_args(obj, custom_args={}):
    """
    Utility function to get the arguments of an object (specified hereafter).
    
    Parameters
    ----------
    obj : class or method
        A python class or method.
    custom_args : (dict, optional)
        A dictonnary containing custom values to be added.

    Returns
    -------
    A dictionnary of the object arguments with their default values.
    """
    # retrieve the dictionnary of arguments
    if not inspect.isclass(obj) and not inspect.isfunction(obj):
        raise TypeError("""'obj' must be either a Python class or a Python
                        function.""")
    else:
        try:
            obj_attr = inspect.getfullargspec(obj)
            start = len(obj_attr[0]) - len(obj_attr[3])
            obj_args = dict(zip(obj_attr[0][start:], obj_attr[3]))
        except TypeError as err:
            print("The object arguments could not be retrieved: {}".format(err))
            return None
    # add custom values
    for key, val in custom_args.items():
        if key in obj_args.keys():
            obj_args[key] = val

    return obj_args


def summary(model, inputs_shape, batch_size, **kwargs):
    """
    Print model summary.

    Parameters
    ----------
    inputs_shape : tuple
        The input tensor shape.
    batch_size : int
        The batch size targeted.

    Returns
    -------
    A table (str) displaying a summary of the model.
    """

    return torchinfo.summary(
        model, (batch_size,) + inputs_shape, **kwargs)