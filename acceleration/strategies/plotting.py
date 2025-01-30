import matplotlib.gridspec as gs
import matplotlib.pyplot as plt
import numpy as np

from msgwam import config
from msgwam.plotting import plot_time_series

from .utils import load_data

def plot_summary(strategy: str) -> None:
    """
    Plot a summary of the integration outputs, including the momentum flux and
    acceleration time series as well as the RMS profiles of each quantity.

    Parameters
    ----------
    strategy
        Configuration strategy for which to plot the integration output.

    """

    widths = [2, 4.5, 0.2]
    fig = plt.figure(constrained_layout=True)
    fig.set_size_inches(1.2 * sum(widths), 1.2 * 6)

    spec = gs.GridSpec(2, 3, figure=fig, width_ratios=widths)
    axes = [[fig.add_subplot(spec[i, j]) for j in range(3)] for i in range(2)]
    axes = np.array(axes)

    fields = ['flux', 'acceleration']
    units = ['mPa', 'm / s']
    factors = [1e3, 86400]
    amaxes = [3, 40]

    zipped = zip(fields, units, factors, amaxes)
    for i, (field, unit, factor, amax) in enumerate(zipped):
        data = factor * load_data('coarse', field, spinup_days=0)
        _, cbar = plot_time_series(data, amax, axes[i, 1:])
        cbar.set_label(f'{field} ({unit})')

        rms = np.sqrt((data ** 2).mean('time'))
        z = data[[s for s in data.coords if s[0] == 'z'][0]] / 1000
        axes[i, 0].plot(rms, z, color='k')

        axes[i, 0].set_xlim(0, amax)
        axes[i, 0].set_ylim(z.min(), z.max())
        axes[i, 0].tick_params('both', direction='in')

        axes[i, 0].grid(color='lightgray')
        axes[i, 0].set_xlabel(f'RMS {field} ({unit})')
        axes[i, 0].set_ylabel('height (km)')

    plt.savefig(f'plots/{config.name}/{strategy}.png', dpi=400)