import cftime
import matplotlib.gridspec as gs
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

from msgwam import config
from msgwam.constants import EPOCH
from msgwam.plotting import plot_time_series
from msgwam.utils import get_vertical_grids, open_dataset

from .coarsening import _get_grid, _get_normalized_errors, _get_path
from .strategies import get_overrides
from .utils import get_rmse, load_data

_COLORS = {
    'coarse' : 'k',
    'stochastic' : 'gold',
    'instantaneous' : 'tab:red',
    # 'surrogate' : 'royalblue'
}

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
    ref = load_data('reference')

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

        rmse = get_rmse(ref, load_data(path))
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

def plot_error_profiles() -> None:
    """
    Plot the RMS errors as a function of height for each strategy.
    """

    names = ['flux', 'acceleration']
    units = ['mPa', 'm / s / day']
    factors = [1000, 86400]

    fig, axes = plt.subplots(ncols=2)
    fig.set_size_inches(6, 4.5)

    zipped = zip(names, units, factors, axes)
    for i, (name, unit, factor, ax) in enumerate(zipped):
        z = get_vertical_grids()[i] / 1000
        ref = load_data('reference', var=name, resample=86400)

        for strategy, color in _COLORS.items():
            data = load_data(strategy, var=name)
            rmse = factor * get_rmse(data, ref)

            ax.plot(rmse, z, color=color, label=strategy)

        rms = factor * get_rmse(ref)
        ax.plot(rms, z, color='gray', ls='dashed', label='RMS')

        xmax = [1, 50][i]
        ax.set_xlim(0, xmax)
        ax.set_ylim(z.min(), z.max())

        ax.set_xlabel(f'{name} RMSE ({unit})')
        ax.set_ylabel('height (km)')

        ax.grid(color='lightgray')
        ax.tick_params('both', direction='in')

    axes[0].legend()
    plt.tight_layout()
    plt.savefig(f'plots/{config.name}/error-profiles.png', dpi=400)

def plot_mean_state() -> None:
    """
    Plot the mean wind time series, along with some profile information about
    the RMS wind scale and the maximum wind at each level.
    """

    widths = [2, 4.5, 0.2]
    fig = plt.figure(constrained_layout=True)
    fig.set_size_inches(sum(widths), 3)

    spec = gs.GridSpec(1, 3, fig, width_ratios=widths)
    axes = [fig.add_subplot(spec[0, j]) for j in range(3)]

    with open_dataset(config.prescribed_wind_file) as ds:
        u = ds['u']

    _, cbar = plot_time_series(u, 60, axes[1:], cmap='PuOr_r')
    cbar.set_label('$\\bar{u}$ (m / s)')

    z = u['z_centers'].values / 1000
    u_max = abs(u).max('time')
    u_rms = get_rmse(u)
    
    axes[0].plot(u_rms, z, color='k', label='RMS')
    axes[0].plot(u_max, z, color='k', ls='dotted', label='max')

    axes[0].set_xlabel('$\\bar{u}$ (m / s)')
    axes[0].set_ylabel('height (km)')
    axes[0].legend()

    axes[0].set_xlim(0, 60)
    axes[0].set_ylim(z.min(), z.max())
    axes[0].tick_params('both', direction='in')
    axes[0].grid(color='lightgray')

    axes[1].set_ylabel(None)
    axes[0].set_yticks(axes[1].get_yticks())
    axes[0].set_yticklabels(axes[1].get_yticklabels())

    plt.savefig(f'plots/{config.name}/descending-jets.png', dpi=400)

def plot_summary(strategy: str) -> None:
    """
    Plot a summary of the integration outputs, including the number of active
    ray volumes as a function of time and the momentum flux time series.

    Parameters
    ----------
    strategy
        Strategy to plot the integration outputs for.

    """

    widths = [4.5, 2, 4.5, 0.2]
    fig = plt.figure(constrained_layout=True)
    fig.set_size_inches(sum(widths), 3)

    spec = gs.GridSpec(
        nrows=1, ncols=len(widths),
        width_ratios=widths,
        figure=fig
    )

    axes = [fig.add_subplot(spec[0, i]) for i in range(3)]
    cax = fig.add_subplot(spec[0, 3])

    with config.override(**get_overrides(strategy)):
        flux = load_data(strategy, spinup_days=0)
        _, cbar = plot_time_series(1000 * flux, 2, [axes[-1], cax])
        cbar.set_label('flux (mPa)')

        z = flux['z_faces'] / 1000
        axes[1].plot(1000 * get_rmse(flux), z, color='k')

        axes[1].set_xlim(0, 1)
        axes[1].set_ylim(z.min(), z.max())
        axes[1].tick_params('both', direction='in')
        axes[1].grid(color='lightgray')

        axes[1].set_xlabel('RMS flux (mPa)')
        axes[1].set_ylabel('height (km)')

        if config.propagator_type == 'transient':
            count = load_data(strategy, 0, None, var='n_rays')

            ymax = 800e3 if strategy == 'reference' else 300
            days = cftime.date2num(flux['time'], f'days since {EPOCH}')
            line = config.n_max * np.ones_like(days)

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