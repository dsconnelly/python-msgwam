import matplotlib.gridspec as gs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from msgwam.plotting import plot_time_series

from msgwam import config

def plot_summaries(
    datas: dict[str, xr.DataArray],
    amaxes: list[float],
    units: list[str]
) -> None:
    """
    Plot time series and RMS values for one or more variables.

    Parameters
    ----------
    datas
        Dictionary mapping field names to `DataArray` objects.
    amaxes
        Maximum (absolute) value to show in plots.
    units
        Units with which to label axes and colorbars.

    """

    n_cols = (len(datas) + 1) // 2
    n_rows = len(datas) // n_cols
    n_x = 3 + 2 * (n_cols - 1)

    fig = plt.figure(constrained_layout=True)
    widths = [2, 4.5] * (n_cols - 1) + [2, 4.5, 0.2]
    fig.set_size_inches(1.2 * sum(widths), 1.2 * 3 * n_rows)

    spec = gs.GridSpec(n_rows, n_x, figure=fig, width_ratios=widths)
    axes = np.empty((n_rows, n_x), dtype=object)

    for i in range(n_rows):
        for j in range(n_x):
            axes[i, j] = fig.add_subplot(spec[i, j])

    zipped = zip(datas.items(), amaxes, units)
    for k, ((field, data), amax, unit) in enumerate(zipped):
        i, j = k % 2, 2 * (k // 2)
        cax = axes[i, -1]

        _, cbar = plot_time_series(data, amax, [axes[i, j + 1], cax])
        cbar.set_label(unit)

        rms = np.sqrt((data ** 2).mean('time'))
        z = np.linspace(config.z_min, config.z_max, data.shape[1]) / 1000
        axes[i, j].plot(rms, z, color='k')

        axes[i, j].set_xlim(0, amax)
        axes[i, j].set_ylim(z.min(), z.max())
        axes[i, j].tick_params('both', direction='in')

        ticks = np.linspace(0, amax, 5)
        axes[i, j].set_xticks(ticks)

        axes[i, j].grid(color='lightgray')
        axes[i, j].set_xlabel(f'RMS {field} ({unit})')
        axes[i, j].set_ylabel('height (km)')