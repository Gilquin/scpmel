# global imports
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch

plt.style.use('paper') # mplstyle

def save_gradflow(model, path):
    """
    Plots the gradients flowing through different layers in the net during
    training. Can be used for checking for possible gradient vanishing or
    exploding problems. It should follow loss.backward() call.

    Parameters
    ----------
    model : torch.nn.Modules
        An instance of a torch.nn.Modules.
    path : str
        A string indicating the path where to save the graph.

    Returns
    -------
    None.

    """
    # disable interactive mode
    plt.ioff()

    # check model
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be an instance of torch.nn.Module.")
    
    # get the gradient values per layer
    ave_grads = []
    max_grads= []
    layers = []
    for idx, (n, p) in enumerate(model.named_parameters()):
        if(p.requires_grad):
            layers.append(idx)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    # plot formatting
    fig, ax = plt.subplots(figsize=(10,8))
    ax.bar(np.arange(len(max_grads)), max_grads, alpha=0.5, lw=1, color="c")
    ax.bar(np.arange(len(max_grads)), ave_grads, alpha=0.5, lw=1, color="b")
    ax.set_xticks(range(0,len(ave_grads), 1))
    ax.set_xticklabels(layers, rotation = "vertical")
    ax.set_xlim(left=-1, right=len(ave_grads))
    ax.set_ylim(bottom = 0) # zoom in on the lower gradient regions
    ax.set_xlabel("Layers")
    ax.set_ylabel("average gradient")
    ax.set_title("Gradient flow")
    ax.grid(True)
    ax.legend([Line2D([0], [0], color="c", lw=4),
               Line2D([0], [0], color="b", lw=4)],
              ['max-gradient', 'mean-gradient'])
    # save the figure
    fig.savefig(path+".png", dpi=200, bbox_inches='tight')
    plt.close(fig)

    
    # reenable interactive mode
    plt.ion()

    return None