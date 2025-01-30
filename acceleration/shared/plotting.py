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

    n_rows, widths = len(datas), [2, 4.5, 0.2]
    fig = plt.figure(constrained_layout=True)

    fig.set_size_inches(1.2 * sum(widths), 1.2 * 3 * n_rows)
    spec = gs.GridSpec(n_rows, 3, figure=fig, width_ratios=widths)
    axes = np.empty((n_rows, 3), dtype=object)

    for i in range(n_rows):
        for j in range(3):
            axes[i, j] = fig.add_subplot(spec[i, j])

    zipped = zip(datas.items(), amaxes, units)
    for i, ((field, data), amax, unit) in enumerate(zipped):
        _, cbar = plot_time_series(data, amax, axes[i, 1:])
        cbar.set_label(f'{field} ({unit})')

        rms = np.sqrt((data ** 2).mean('time'))
        z = np.linspace(config.z_min, config.z_max, data.shape[1]) / 1000
        axes[i, 0].plot(rms, z, color='k')

        axes[i, 0].set_xlim(0, amax)
        axes[i, 0].set_ylim(z.min(), z.max())
        axes[i, 0].tick_params('both', direction='in')

        axes[i, 0].grid(color='lightgray')
        axes[i, 0].set_xlabel(f'RMS {field} ({unit})')
        axes[i, 0].set_ylabel('height (km)')