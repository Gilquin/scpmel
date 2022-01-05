# global imports
import numpy as np
import torch
# relative import
from .. import utils

class SubspaceOptimizer(object):
    """
    Optimizer that performs a stochastic gradient descent on a subspace of the
    parameters defined by a low rank matrix resulting from a dimensionality
    reduction method (for instance a Singular Value Decomposition (SVD)). The 
    low rank matrix is expected to have shape (p, r) where:
        * p is the number of model parameters,
        * r is the rank approximation of the matrix, e.g. the number of principal
        components spawning the subspace.

    Parameters
    ----------
    optimizer: torch.nn.optim
        One of a torch torch.optim class.
    parameters : nn.Modules.named_parameters
        A generator of the model named parameters: model.named_parameters().
    opt_args : dict
        A dict containing custom arguments of the torch.optim class.
    Mat : torch.tensor
        The low rank matrix used to map the model parameters into a subspace.
    avg : torch.tensor
        The vector of parameter averages upon the samples used to obtain the low
        rank matrix. It is used to construct the shift vector.
    rank : int (default None)
        The rank of the matrix. Default to None, as the matrix "Mat" is expected 
        to alredady be in truncated form. If "Mat" is given in compact form, 
        the value passed to rank will be used to discard the remaining rows. 
    """

    def __init__(self, optimizer, parameters, opt_args, Mat, avg, rank=None,
                 device="cpu"):
        
        # get trainable parameters
        self._params, self._params_map = self._get_trainable_parameters(parameters)
        self._nparams = sum([param.numel() for param in self._params.values()])
        # get the low rank matrix
        Mat, self.rank = self._check_matrix(Mat, rank)
        # construct the shift vector
        shift = self._get_shift(Mat, avg)
        # send to correct device
        self.Mat = Mat.to(device)
        self.shift = shift.to(device)
        # instantiate the optimize with principal components
        opt_args = utils.get_args(optimizer, opt_args)
        pc = torch.zeros((self.rank), device=device)
        pc.grad = torch.zeros((self.rank), device=device)
        self.optimizer = optimizer((pc,), **opt_args)


    def _check_matrix(self, Mat, rank):
        """
        Check wheter the matrix is in truncated or compact form and with the correct
        shape.
        """
        if rank is not None:
            Mat = Mat[...,:rank]
            rank = rank
        elif Mat.shape[0] != self._nparams: # transpose
            Mat = Mat.T
            rank = Mat.shape[1]
        return Mat, rank


    def _get_shift(self, Mat, avg):
        """
        Construct the shift vector.
        """
        return (torch.diag(torch.ones(Mat.shape[0])) - Mat @ Mat.T) @ avg


    def _get_trainable_parameters(self, parameters):
        """
        Store trainable parameters in an OrderedDict and construct a mapping 
        of the trainable parameters range per layer.
        """
        params_dict = dict({
            param[0]:param[1] for param in filter(lambda p: p[1].requires_grad, parameters)})
        # get a map of the trainable parameters range per layer
        with torch.no_grad():
            param_counts = np.cumsum([0]+[*map(torch.numel, params_dict.values())])
            param_ranges = map(lambda x: range(*x), zip(param_counts[:-1], param_counts[1:]))
            params_map = dict([*zip([*params_dict.keys()], [*param_ranges])])
        return params_dict, params_map


    def map_to_subspace(self):
        """
        Map the parameters values and gradients to the subspace spawn by the
        principal components.
        """
        # get the principal components values
        param_vals = torch.cat([param.flatten() for param in self._params.values()])
        self.optimizer.param_groups[-1]["params"][0].copy_(self.Mat.T @ (param_vals - self.shift))
        # get the principal components gradients
        grad_vals = torch.cat([param.grad.flatten() for param in self._params.values()])
        self.optimizer.param_groups[-1]["params"][0].grad.copy_(self.Mat.T @ grad_vals)
        return None

    
    def map_from_subspace(self):
        """
        Map the principal components values and gradients back to the space spawn
        by the model parameters.
        """
        for name, param in self._params.items():
            param_range = self._params_map[name]
            param_vals = self.shift[param_range] + self.Mat[param_range, :] @ self.optimizer.param_groups[-1]["params"][0]
            param.copy_(param_vals.view_as(param))
            param_grads = self.Mat[self._params_map[name], :] @ self.optimizer.param_groups[-1]["params"][0].grad
            param.grad.copy_(param_grads.view_as(param))
        return None


    @torch.no_grad()
    def step(self):
        # get the principal components values and gradients
        self.map_to_subspace()
        # apply the optimizer step on the subspace
        self.optimizer.step()
        # get the new parameters values and gradients
        self.map_from_subspace()
        return None

    def zero_grad(self, set_to_None=False):
        self.optimizer.zero_grad(set_to_None)
        return None

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)