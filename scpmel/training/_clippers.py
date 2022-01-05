"""
Provides clipping class to restrict weights of a layer after a gradient update.
This is currently the only way to "apply" constraints to the weights in Pytorch.
The only alternative is to clip the gradient update that is modifying the
optimizer.

The clipping is applied after the optimization step:
    
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = model.loss(outputs, targets)
    loss.backward()
    optimizer.step()

    if epoch % clipper.frequency == 0:
        model.apply(clipper)

The clipper frequency controls the clipping rate per epoch, a value of 1 means
clipping at each batch. 

Reference links:
    https://discuss.pytorch.org/t/restrict-range-of-variable-during-gradient-descent/1933/3
    https://discuss.pytorch.org/t/set-constraints-on-parameters-or-layers/23620
"""

# global import
import torch

__all__ = ["UnitNormClipper", "ZeroOneClipper", "ZeroSumClipper",
           "UnitSumClipper"]


class _Clipper(object):
    """
    Parent class for all clippers.
    """
    def __init__(self, frequency=1):
        self.frequency = frequency

    def _check_module(self, module):
        cond = isinstance(module, torch.nn.Module) and\
        hasattr(module, 'weight') and module.weight.requires_grad
        return cond
        

class UnitNormClipper(_Clipper):
    """
    Normalized all the weights of each module.
    """
    def __init__(self, frequency):
        super().__init__(frequency)

    def __call__(self, module):
        if self._check_module(module):
            w = module.weight.data
            w.div_(torch.norm(w, 2, -1)[...,None])


class ZeroOneClipper(_Clipper):
    """
    Restrict the weights of each module to be in the range [0,1].
    """
    def __init__(self, frequency):
        super().__init__(frequency)

    def __call__(self, module):
        if self._check_module(module):
            w = module.weight.data
            m = torch.min(w, -1)[0]
            M = torch.max(w, -1)[0]
            w.sub_(m[...,None]).div_((M-m)[...,None])


class ZeroSumClipper(_Clipper):
    """
    Restrict the weights of each module to sum to zero by substracting the
    sum to each weight.
    """
    def __init__(self, frequency=5):
        super().__init__(frequency)

    def __call__(self, module):
        if self._check_module(module):
            w = module.weight.data
            w.sub_(torch.sum(w, -1)[...,None]/w.shape[-1])


class UnitSumClipper(_Clipper):
    """
    Restrict the weights of each module to sum to 1 by diviving each
    weight by the sum.
    """
    def __init__(self, frequency=5):
        super().__init__(frequency)

    def __call__(self, module):
        if self._check_module(module):
            w = module.weight.data
            w.div_(torch.sum(w, -1)[...,None])


if __name__ == '__main__':
    # print module info
    def module_infos(module):
        if hasattr(module, "weight"):
            print("module name: {}\n".format(type(module)))
            print("weights: {}".format(module.weight.data))
    # test net
    net = torch.nn.Sequential(
        torch.nn.Conv1d(1, 4, 3),
        torch.nn.ReLU(),
        torch.nn.BatchNorm1d(4),
        torch.nn.Conv1d(4, 2, 5)
        )
    # pass trough random data
    out = net(torch.rand((1,1,10), dtype=torch.get_default_dtype()))
    # print current sets of weights
    net.apply(module_infos)
    # test clippers
    print("After ZeroOneClipper:\n")
    clipper = ZeroOneClipper(1)
    net.apply(clipper)
    net.apply(module_infos)
    print("After UnitNormClipper:\n")
    clipper = UnitNormClipper(1)
    net.apply(clipper)
    net.apply(module_infos)
    print("After UnitSumClipper:\n")
    clipper = UnitSumClipper(1)
    net.apply(clipper)
    net.apply(module_infos)
    print("After ZeroSumClipper:\n")
    clipper = ZeroSumClipper(1)
    net.apply(clipper)
    net.apply(module_infos)