# global imports
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d # necessary for 3d plots
import numpy as np

plt.style.use('paper') # mplstyle


def _int_to_str(val, length=4):
    """
    Zero padding of index to ease postprocessing of images.
    """
    out = str(val)
    while len(out) < length:
        out = '0' + out
    return out


def tri_plot(solutions, xx, tt, ylims, title=None):
    """
    Vertical alignement of three 1D plot with shared axis. 

    Parameters
    ----------
    solutions : list
        A list containing the three solutions to be plotted.
    xx : array like
        The spatial points.
    tt : array like
        The corresponding times.
    ylims : list
        The shared y-axis limits.
    title : str
        The main title of the figure.

    Returns
    -------
    The matplotlib figure.

    """

    fig, ax = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(8,9))
    plt.setp(ax, ylim=ylims)
    for pos in range(3):
        ax[pos].plot(xx, solutions[pos], label="time: {} s".format(tt[pos]))
        ax[pos].axhline(0, linestyle='--', linewidth=0.5, c="black")
        ax[pos].legend(fontsize=14, loc='lower left')
    if title is not None:
        plt.suptitle(title, fontsize=18)
    plt.tight_layout()

    return fig



def surface_plot(solutions, xx, tt):
    """
    Uni or dual (x,t) surface plot of solutions.

    Parameters
    ----------
    solutions : dict
        A dictionary containing the solution(s), array-like, to be plotted.
    xx : np.ndarray
        The spatial points.
    tt : np.meshgrid
        The time steps.

    Returns
    -------
    The matplotlib figure.

    """

    # number of solutions (1 or 2)
    nb_sol = len(solutions.keys())

    # meshgrid
    xx, tt = np.meshgrid(xx, tt)

    # plot in time-space domain the solution(s)
    fig = plt.figure()
    for idx, label in enumerate(solutions.keys()):
        ax = fig.add_subplot(1, nb_sol, idx+1, projection='3d')
        zz = solutions[label]
        norm = plt.Normalize(zz.min(), zz.max())
        colors = cm.viridis(norm(zz))
        surf = ax.plot_surface(xx, tt, zz, facecolors=colors, shade=False)
        surf.set_facecolor((0,0,0,0))

    plt.show()
    
    return fig


def export_as_gif(solutions, xx, tt, path):
    """
    Uni or dual plot of solutions.

    Parameters
    ----------
    solutions : dict
        A dictionary containing the solution(s), as np.ndarray, to be plotted.
    xx : np.ndarray
        The space grid.
    tt : np.meshgrid
        The time grid.
    path : str
        The folder path to export the collection of plots as pngs.

    Returns
    -------
    The matplotlib figure.
    """

    # disable interactive mode
    plt.ioff()

    # iterate over time steps
    order = int(np.ceil(np.log10(len(tt))))
    labels = list(solutions.keys())

    lims = (
        min([solutions[key].min() - 0.01 for key in labels]),
        max([solutions[key].max() + 0.01 for key in labels])
        )

    for idx in range(len(tt)):
        print("k: {}".format(idx))

        # format time
        time = tt[idx]
        day = int(time // (3600*24))
        resid = time%(3600*24)
        hours = int(resid // 3600)
        minutes = int(resid - hours*3600) // 60
        seconds = int(resid - hours*3600 - minutes*60)
        time_str = 't={}d {}h {}m {}s'.format(day,hours,minutes,seconds)

        # custom figure
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        ax.set_xlabel('spatial domain (x)', fontsize=20)

        # handle uni case
        if len(labels) == 1:
            ax.plot(xx, solutions[labels[0]][idx])
            ax.set_ylim(*lims)
            ax.tick_params(axis='both', labelsize=20)
            ax.set_title(labels[0] + ' at ' + time_str, fontsize=20)

        # handle dual case
        if len(labels) == 2:
            # Turn off axis lines and ticks of the big subplot
            ax.spines['top'].set_color('none')
            ax.spines['bottom'].set_color('none')
            ax.spines['left'].set_color('none')
            ax.spines['right'].set_color('none')
            ax.tick_params(labelcolor='w', top=False, bottom=False, left=False,
                           right=False)
            # add secondaries axes
            ax1 = fig.add_subplot(211)
            ax2 = fig.add_subplot(212)
            # get the solutions
            sol_a = solutions[labels[0]][idx]
            sol_b = solutions[labels[1]][idx]
            # draw graphs
            ax1.plot(xx, sol_a, linewidth=1)
            ax1.set_ylim(*lims)
            ax1.tick_params(axis='both', labelsize=20)
            ax2.plot(xx, sol_b, linewidth=1)
            ax2.set_ylim(*lims)
            ax2.tick_params(axis='both', labelsize=20)
            ax1.set_title(labels[0] + ' at ' + time_str, fontsize=20)
            ax2.set_title(labels[1] + ' at ' + time_str, fontsize=20)
            # adjust spacing
            plt.subplots_adjust(hspace=0.5)

        # export options
        plt.tight_layout()
        plt.savefig(path+'{}.png'.format(_int_to_str(idx, order)),
                    transparent=False, dpi=200)
        # close figure
        plt.close()

    # reenable interactive mode
    plt.ion()

    return None