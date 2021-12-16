# global imports
import matplotlib.pyplot as plt
plt.style.use('paper') # mplstyle

def LR_plot(loss_data, lr_data, **kwargs):
    """
    Draw the loss and learning rate data on two y-axes that share the same x-axis.

    Parameters
    ----------
    loss_data : array-like or scalar
        The loss data to plot on the first y-axis.
    lr_data : array-like or scalar
        The lr data to plot on the second y-axis.

    Returns
    -------
    The matplotlib figure.
    """

    fig = plt.figure(figsize=(8,6))
    ax1 = fig.add_subplot(1,1,1)
    ax1.set_xlabel('epoch')

    ax1.set_ylabel('loss')
    ax1.plot(loss_data, color="tab:blue")
    ax1.tick_params(axis='y')
    ax1.set_yscale("log")

    ax2 = ax1.twinx() # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('lr', rotation=-90)  # we already handled the x-label with ax1
    ax2.plot(lr_data, color="tab:orange", linestyle="--")
    ax2.tick_params(axis='y')
    ax2.set_yscale("log")

    fig.tight_layout() # otherwise the right y-label is slightly clipped
    fig.show()

    return fig
