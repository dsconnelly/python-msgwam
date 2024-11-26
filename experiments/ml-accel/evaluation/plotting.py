import cftime
import matplotlib.gridspec as gs
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

from msgwam import config
from msgwam.constants import EPOCH
from msgwam.plotting import plot_source, plot_time_series
from msgwam.utils import get_vertical_grids

from .coarsening import _get_grid, _get_normalized_errors, _get_path
from .strategies import get_overrides
from .utils import get_rmse, load_data

def plot_coarse_errors() -> None:
    """
    Plot the normalized error for each coarsened integration, and plot the RMSE
    as a function of height for the best and worst configurations.
    """

    widths = [3, 4.5, 0.2]
    fig = plt.figure(constrained_layout=True)
    fig.set_size_inches(sum(widths), 4.5)

    spec = gs.GridSpec(1, 3, figure=fig, width_ratios=widths)
    axes = [fig.add_subplot(spec[0, j]) for j in range(2)]
    cax = fig.add_subplot(spec[0, 2])

    width = 1
    colors = ['darkgreen', 'w', 'darkred']
    cmap = LinearSegmentedColormap.from_list('custom', colors, 256)

    drs, n_sources = _get_grid()
    errors = _get_normalized_errors()

    img = axes[1].imshow(
        errors.T,
        vmin=(1 - width),
        vmax=(1 + width),
        origin='lower',
        aspect='auto',
        cmap=cmap
    )

    dcs = [round(_get_dc(n), 2) for n in n_sources]
    axes[1].set_xticks(np.arange(len(drs)), labels=drs, rotation=45)
    axes[1].set_yticks(np.arange(len(n_sources)), labels=dcs)

    axes[1].set_xlabel('$\\delta z$ (m)')
    axes[1].set_ylabel('$\\delta c_{\mathrm{p}}$ (m / s)')

    cbar = plt.colorbar(img, cax=cax)
    cbar.set_ticks(np.linspace(1 - width, 1 + width, 5))
    cbar.set_label('normalized error')

    z = get_vertical_grids()[0] / 1000
    ref = load_data('reference', 5, '3h')

    funcs = [np.argmin, np.argmax]
    colors = ['darkgreen', 'darkred']
    labels = ['best', 'worst']

    for func, color, label in zip(funcs, colors, labels):
        i, j = np.unravel_index(func(errors), errors.shape)
        path = _get_path(drs[i], n_sources[j])

        axes[1].add_patch(Rectangle(
            (i - 0.5, j - 0.5), 1, 1,
            ec=color, fc='none',
            clip_on=False,
            linewidth=2,
            zorder=10
        ))

        rmse = get_rmse(ref, load_data(path, 5, '3h'))
        axes[0].plot(1000 * rmse, z, color=color, label=label)

    rms = 1000 * get_rmse(ref)
    axes[0].plot(rms, z, color='gray', ls='dashed', label='RMS flux')

    axes[0].set_xlim(0, 2)
    axes[0].set_ylim(z.min(), z.max())
    axes[0].tick_params('both', direction='in')
    axes[0].grid(color='lightgray')

    axes[0].set_xlabel('RMSE (mPa)')
    axes[0].set_ylabel('height (km)')
    axes[0].legend()
    
    plt.savefig(f'plots/{config.name}/coarse-errors.png', dpi=400)

def plot_summary(strategy: str) -> None:
    """
    Plot a summary of the integration outputs, including the number of active
    ray volumes as a function of time and the momentum flux time series.

    Parameters
    ----------
    strategy
        Strategy to plot the integration outputs for.

    """

    widths = [4.5, 4.5, 0.2]
    fig = plt.figure(constrained_layout=True)
    fig.set_size_inches(sum(widths), 3)

    spec = gs.GridSpec(
        nrows=1, ncols=3,
        width_ratios=widths,
        figure=fig
    )

    axes = [fig.add_subplot(spec[0, i]) for i in range(2)]
    cax = fig.add_subplot(spec[0, 2])

    with config.override(**get_overrides(strategy)):
        count = load_data(strategy, var='n_rays')
        flux = load_data(strategy)

        ymax = config.n_max + 10 ** np.floor(np.log10(config.n_max))
        days = cftime.date2num(count['time'], f'days since {EPOCH}')
        line = config.n_max * np.ones_like(days)

        _, cbar = plot_time_series(1000 * flux, 3, [axes[1], cax])
        cbar.set_label('flux (mPa)')

    axes[0].plot(days, count, color='k')
    axes[0].plot(days, line, color='gray', ls='dashed')

    axes[0].set_xlim(days.min(), days.max())
    axes[0].set_xlabel('time (days)')

    axes[0].set_ylim(0, ymax)
    axes[0].set_yticks(np.linspace(0, ymax, 5))
    axes[0].set_ylabel('active ray volumes')

    axes[0].grid(color='lightgray')
    axes[0].tick_params('both', direction='in')

    plt.savefig(f'plots/{config.name}/{strategy}-summary.png', dpi=400)
    plot_source(f'plots/{config.name}/{strategy}-source.png')

def _get_dc(n: int) -> float:
    """
    Get the phase velocity resolution corresponding to a particular number of
    spectral elements at the source.

    Parameters
    ----------
    n
        Number of spectral source elements

    Returns
    -------
    float
        Extent of each source ray volume in phase velocity space.

    """

    return np.diff(np.linspace(-config.c_max, config.c_max, n + 1))[0]