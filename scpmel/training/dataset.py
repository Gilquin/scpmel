# global imports
import collections.abc as container_abcs
import h5py
import numpy as np
import torch
from torch.utils.data import random_split
from torch.utils.data._utils.collate import default_collate
# relative import
from .. import les

__all__ = ["custom_collate_fn", "splitter", "H5Dataset"]


def custom_collate_fn(batch):
    """
    This custom version handles empty tuples by discarding them from the
    batch. As a result batch size will differ from a few elements.
    """

    elem = batch[0]
    if isinstance(elem, container_abcs.Sequence):
        it = iter(batch)
        # remove empty tuple from the batch
        batch = [elem for elem in it if len(elem)>0]

    return default_collate(batch)


def splitter(dataset, valid_split=0.1):
    """
    Utility to split a dataset into training and validation subsets.

    Parameters
    ----------

    dataset : map-style or iterable-style dataset.
        The dataset from which to load the data. Its type must be compatible
        with the data_loader, for more infos see:
            https://pytorch.org/docs/stable/data.html#map-style-datasets
    valid_split : (float, optional)
        A float between 0 and 1 denoting the ratio of the dataset used for
        validation.

    Returns
    -------
    The training and validation subset index.
    """

    # get training and validation subset sizes
    train_size = int(len(dataset) * (1-valid_split))
    if train_size%2 != 0: # ensure to get an even size for the training dataset
        train_size -= 1
    valid_size = len(dataset) - train_size

    # random splitting
    train_idx, valid_idx = random_split(
        range(len(dataset)), [train_size, valid_size])

    return train_idx, valid_idx


class H5Dataset(torch.utils.data.Dataset):
    """
    Utility class that defines a Dataset class compatible with a
    torch.utils.data.DataLoader instance. Users should pay attention to the
    following recommendations: (see
    https://discuss.pytorch.org/t/dataloader-when-num-worker-0-there-is-bug/25643
    for an in depth discussion)
        * use HDF5 in version 1.10,
        * use DataLoader with arguments 'num_workers > 0' and batch_sampler
    
    The returned tuple is a triple of (input, target, index) formatted to agree
    with the default automatic batching, see the link below for more
    informations:
        https://pytorch.org/docs/stable/data.html#automatic-batching-default
    In particular the automatic batching add the batch dimension as the first
    dimension, as such data should have the shape:
        (in_channels, (...))
    where (...) denotes the spatial dimensions (e.g. 1D, 2D or 3D).
    """

    def __init__(self, path, name="dataset"):
        self.file_path = path
        self.name = name
        self.dataset = None
        self.subsampl = None
        self.time_scale = None
        self.mode = None
        self.dataset_len = None
        self.keys = None
        with h5py.File(self.file_path, 'r', libver="latest") as file:
            self.filter = file[self.name]["filter"][:]
            self.subsampl = int(file[self.name].attrs["ratio"] / 2)

    def set_mode(self, mode):
        with h5py.File(self.file_path, 'r', libver="latest") as file:
            if mode == "training":
                self.keys = np.array(list(file[self.name]["training_keys"]))
            elif mode == "validation":
                self.keys = np.array(list(file[self.name]["validation_keys"]))
            else:
                raise KeyError(
                    """
                    Invalid argument 'mode', it must be one of {'training',
                    'validation'}.
                    """)
        self.dataset_len = len(self.keys)
        self.mode = mode

    def __getitem__(self, index):
        # check that keys is defined
        if self.keys is None:
            raise AttributeError(
                """
                Argument 'mode' has not been set, use the "set_mode" method.
                """)
        # gimmick for distributed sampler to work
        if self.dataset is None:
            self.dataset = h5py.File(
                self.file_path, 'r', libver='latest')["data"]
        # get tuple of keys
        keys = self.keys[index]
        # apply filter to each data
        data = les.filtr(
            self.dataset[keys[0]][:], self.filter)
        target = les.filtr(
            self.dataset[keys[1]][:], self.filter)
        # return tuple (input, target)
        return (data[0,...,::self.subsampl], target[0,...,::self.subsampl], index)

    def __len__(self):
        if self.dataset_len is None:
            raise AttributeError(
                """
                Argument 'mode' has not been set, use the "set_mode" method.
                """)
        return self.dataset_len

    def __del__(self):
        try:
            self.dataset.file.close()
        except AttributeError:
            pass
            # print("Excepting error {}: the file has already been closed.".format(err))


if __name__ == '__main__':
    # test H5Dataset container
    import os
    import random
    path = "../../examples/KDV/data/KDV_LES_800.0.hdf5"
    path = os.path.join(os.path.dirname(__file__), path)
    dset = H5Dataset(path=path,name="cutoff1")
    dset.set_mode("training")
    # test random key
    key = random.randint(0, len(dset.keys)-1)
    out = dset[key]
    # test empty tuple
    key = len(dset.keys)-1
    out = dset[key]

    print("Dataset infos:")
    print("\tfile instance: {}".format(dset.dataset.file))
    print("\trelative path to root: {}".format(dset.dataset.name))
    print("\tnumber of objects contained: {}".format(dset.dataset.id.get_num_objs()))
    print("\tattributes:\n\t" + str({**dset.dataset.attrs}))

    # instantiate DataLoader with custom collate function
    loader = torch.utils.data.DataLoader(
        dset, batch_size=100, shuffle=True, pin_memory=True, sampler=None)
    print("loader length: {}".format(len(loader)))
    # iterate over batch
    for batch_idx, batch in enumerate(loader):
        inputs, targets, index = batch
        print("batch indexed {} contains {} elements".format(
            batch_idx, len(inputs)))