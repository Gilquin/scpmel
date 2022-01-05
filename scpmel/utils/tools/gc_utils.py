# global imports
import gc
import os
import psutil
import sys
import torch

def _checkfile(fname, overwrite):
    if os.path.isfile(fname) and not overwrite:
        action = "a"
    else:
        action = "w"
    return action

def cpuStats(fname, overwrite=False):
    # cpu usage infos
    action = _checkfile(fname, overwrite)
    with open(fname, action) as f:
        print(sys.version, file=f)
        print(psutil.cpu_percent(), file=f)
        print(psutil.virtual_memory(), file=f)  # physical memory usage
        pid = os.getpid()
        py = psutil.Process(pid)
        memoryUse = py.memory_info()[0] / 2. ** 20 # memory use in MB
        print('memory MB:', memoryUse, file=f)
    return None

def memProc(fname, overwrite=False):
    # cpu usage infos
    action = _checkfile(fname, overwrite)
    with open(fname, action) as f:
        pid = os.getpid()
        py = psutil.Process(pid)
        memoryUse = py.memory_info()[0] / 2. ** 20 # memory use in MB
        print(memoryUse, file=f)
    return None

def memReport(fname, overwrite=False):
    # garbage collector infos
    action = _checkfile(fname, overwrite)
    with open(fname, action) as f:
        print("gc count: {}".format(gc.get_count()), file=f)
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)): 
                    print(type(obj), obj.size(), file=f)
            except:
                pass
    return None