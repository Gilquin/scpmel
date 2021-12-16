# global imports
import torch
from torch.nn.modules.loss import _Loss

__all__ = ["get_loss", "KineticLoss"]


def get_loss(loss_func, silent=True):
    """
    Method that instantiates the loss function used for training the model.

    Parameters
    ----------
    loss_func : string or class
        Either a string to select a pytorch.nn loss function:
            https://pytorch.org/docs/stable/nn.html#loss-functions
        or a custom loss function. Note that the custom function implementation
        must follow some rules, see some examples at:
            https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch

    Returns
    -------
    An instance of a torch.nn.modules.loss.

    """

    # instantiate loss function
    if isinstance(loss_func, str):
        loss_func = getattr(torch.nn, loss_func)(reduction='mean')
    else:
        if not issubclass(type(loss_func), _Loss):
            raise TypeError("""'loss_func' must be a class that inherits from
                            torch.nn.modules.loss.""")
        elif not hasattr(loss_func, 'forward'):
            raise AttributeError("""'loss_func' must implement its owns
                                 forward method.""")

    if not silent:
        print("\nLoss function: \n\t{}".format(str(loss_func)[:-2]))

    return loss_func


# kinetic loss function
class KineticLoss(_Loss):
    """
    Loss derived from the resolved kinetic energy.
    """

    def __init__(self, reduction='mean'):
        super(KineticLoss, self).__init__(None, None, reduction)
        if self.reduction not in ["none", "sum", "mean"]:
            raise ValueError("{} is not a valid value for reduction".format(reduction))

    def forward(self, outputs, targets):
        res = torch.abs((outputs**2).mean(tuple(range(2, targets.ndim))) -\
                        (targets**2).mean(tuple(range(2, targets.ndim)))).sum(axis=-1)
        if self.reduction == "mean":
            return torch.mean(res)
        elif self.reduction == "sum":
            return torch.sum(res)
        else:
            return res


# TODO: Once FinDiff has been included change that
# H1 loss function
# class H1Loss(_Loss):

#     def __init__(self, dx, order, reduction='mean'):
#         super(H1Loss, self).__init__(None, None, reduction)
#         if self.reduction not in ["none", "sum", "mean"]:
#             raise ValueError("{} is not a valid value for reduction".format(reduction))
#         kernel = get_stencil(dx, order)
#         self.derivative = torch_utils.ConvLayer(kernel)

#     def forward(self, outputs, targets):
#         diff = outputs - targets
#         grad = self.derivative(diff)
#         res = (diff)**2 + grad**2
#         if self.reduction == "mean":
#             return torch.mean(res)
#         elif self.reduction == "sum":
#             return torch.sum(res)
#         else:
#             return res