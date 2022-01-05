# global imports
import torch

__all__ = ["reset_params", "zero_init_params"]


def reset_params(module):
    """
    Reset the trainable parameters of a layer by calling the default method of
    the module. Remark that all parameters need to have their "requires_grad" set to True.
    This function does not handle edge cases, for instance a convolutional layer with
    trainable weights and fixed biases.
    """
    reset_parameters = getattr(module, "reset_parameters", None)
    if callable(reset_parameters) and all(map(lambda param: param.requires_grad, module.parameters())):
        module.reset_parameters()
    return None


def zero_init_params(module):
    """
    Zero initialize the trainable parameters of a layer.
    """
    reset_parameters = getattr(module, "reset_parameters", None)
    if callable(reset_parameters):
        for param in filter(lambda param: param.requires_grad, module.parameters()):
            torch.nn.init.zeros_(param)
    return None