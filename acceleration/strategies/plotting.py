import matplotlib.gridspec as gs
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

from msgwam import config
from msgwam.utils import get_vertical_grids

from ..shared.plotting import plot_summaries

from .coarsenings import get_coarse_errors
from .utils import get_rmse, load_data

_COLORS = {
    'coarse' : 'k',
    'instantaneous' : 'tab:red',
    'stochastic-1' : 'royalblue',
    'stochastic-25' : 'forestgreen'
}

def plot_coarse_errors() -> None:
    """
    Plot the normalized error and RMSE as a function of height for integration
    with coarse resolution.
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

    ref = load_data('reference')
    profiles = get_coarse_errors(ref)

    rms = get_rmse(ref)
    errors = (profiles / rms).mean('z_faces')

    drs = errors['dr'].values
    n_sources = errors['n_source'].values
    dcs = [round(_get_dc(n), 2) for n in n_sources]
    z = get_vertical_grids()[0] / 1000

    img = axes[1].imshow(
        errors.values.T,
        vmin=(1 - width),
        vmax=(1 + width),
        origin='lower',
        aspect='auto',
        cmap=cmap
    )

    axes[1].set_xticks(np.arange(len(drs)), labels=drs, rotation=45)
    axes[1].set_yticks(np.arange(len(n_sources)), labels=dcs)    

    axes[1].set_xlabel('$\\delta z$ (m)')
    axes[1].set_ylabel('$\\delta c_{\\mathrm{p}}$ (m / s)')

    cbar = plt.colorbar(img, cax=cax)
    cbar.set_ticks(np.linspace(1 - width, 1 + width, 5))
    cbar.set_label('normalized error')

    funcs = [errors.argmin, errors.argmax]
    colors = ['darkgreen', 'darkred']
    labels = ['best', 'worst']

    extrema = {}
    for func, color, label in zip(funcs, colors, labels):
        i, j = (da.item() for da in func(...).values())
        extrema[(i, j)] = (color, label, 1, 10)

        axes[1].add_patch(Rectangle(
            (i - 0.5, j - 0.5), 1, 1,
            ec=color, fc='none',
            clip_on=False,
            linewidth=2,
            zorder=10
        ))
    
    defaults = ('k', None, 0.2, 1)
    for i in range(profiles.shape[0]):
        for j in range(profiles.shape[1]):
            profile = 1000 * profiles[i, j].values
            color, label, alpha, zorder = extrema.get((i, j), defaults)

            axes[0].plot(
                profile, z,
                color=color,
                label=label,
                alpha=alpha,
                zorder=zorder
            )

    axes[0].plot(1000 * rms.values, z, color='k', ls='dashed', label='RMS')

    axes[0].set_xlim(0, 2)
    axes[0].set_ylim(z.min(), z.max())
    axes[0].tick_params('both', direction='in')

    axes[0].grid(color='lightgray')
    axes[0].set_axisbelow(True)

    axes[0].set_xlabel('RMSE (mPa)')
    axes[0].set_ylabel('height (km)')
    axes[0].legend()

    plt.savefig(f'plots/{config.name}/coarsenings.png', dpi=400)

def plot_error_profiles(*strategies: str) -> None:
    """Plot RMS errors as a function of height for each strategy."""

    fields = ['flux', 'acceleration']
    units = ['mPa', 'm / s / day']
    factors = [1000, 86400]

    fig, axes = plt.subplots(ncols=2)
    fig.set_size_inches(6, 4.5)

    zipped = zip(fields, units, factors, axes)
    for i, (field, unit, factor, ax) in enumerate(zipped):
        z = get_vertical_grids()[i] / 1000
        ref = load_data('reference', field)

        for strategy in strategies:
            data = load_data(strategy, field)
            rmse = factor * get_rmse(data, ref)
            ax.plot(rmse, z, color=_COLORS[strategy], label=strategy)

        rms = factor * get_rmse(ref)
        ax.plot(rms, z, color='gray', ls='dashed', label='RMS')

        ax.set_xlim([0, 1e-1][i], [2, 50][i])
        ax.set_ylim(config.z_min / 1e3, config.z_max / 1e3)

        if i == 1:
            ax.set_xscale('log')

        ax.set_xlabel(f'{field} RMSE ({unit})')
        ax.set_ylabel('height (km)')

        ax.grid(color='lightgray')
        ax.tick_params('both', direction='in')

    axes[0].legend()
    plt.tight_layout()
    plt.savefig(f'plots/{config.name}/errors.png', dpi=400)

def plot_strategy(strategy: str) -> None:
    """
    Plot a summary of the integration outputs, including the momentum flux and
    acceleration time series as well as the RMS profiles of each quantity.

    Parameters
    ----------
    strategy
        Configuration strategy for which to plot the integration output.

    """

    zipped = zip(['flux', 'acceleration'], [1e3, 86400])
    datas = {s : f * load_data(strategy, s, 0) for s, f in zipped}
    plot_summaries(datas, amaxes=[3, 40], units=['mPa', 'm / s / day'])
    plt.savefig(f'plots/{config.name}/{strategy}.png', dpi=400)

def _get_dc(n: int) -> float:
    """
    Calculate the phase velocity resolution corresponding to a particular number
    of spectral elements at the source.

    Parameters
    ----------
    n
        Number of spectral elements.

    Returns
    -------
    float
        Extent of each source ray volume in phase velocity space.

    """

    return np.diff(np.linspace(-config.c_max, config.c_max, n + 1))[0]