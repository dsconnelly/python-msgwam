import sys

import cftime
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import numpy as np
import xarray as xr

sys.path.insert(0, '.')
from msgwam import config
from msgwam.constants import EPOCH
from msgwam.plotting import plot_source, plot_time_series
from msgwam.utils import open_dataset

import strategies

COLORS = {
    'ICON' : 'k',
    'do-nothing' : 'gold',
    'coarse-square' : 'forestgreen',
    'coarse-tall' : 'royalblue',
    'coarse-wide' : 'tab:red',
    'stochastic' : 'darkviolet',
    'network' : 'fuchsia'
}

def plot_sources() -> None:
    """Plot the source spectrum for each strategy."""

    n_plots = len(COLORS) + 1
    n_cols = int(n_plots // 2)
    n_cols = n_cols + n_plots % 2
    n_rows = 2

    widths = [4.5] * n_cols + [0.2]
    fig = plt.figure(constrained_layout=True)
    fig.set_size_inches(sum(widths), 3 * n_rows)

    grid = gs.GridSpec(
        nrows=n_rows, ncols=(n_cols + 1),
        width_ratios=widths,
        figure=fig
    )

    axes = [fig.add_subplot(grid[i // 4, i % 4]) for i in range(n_plots)]
    cax = fig.add_subplot(grid[:, -1])
    amax = 0.4

    setups = ['reference'] + list(COLORS.keys())
    for setup, ax in zip(setups, axes):
        getattr(strategies, '_' + setup.replace('-', '_'))()
        img = plot_source(amax, [ax])
        config.reset()

        ax.set_title(_format(setup))

    cbar = plt.colorbar(img, cax=cax)
    cbar.set_label('momentum flux (mPa)')

    plt.savefig(f'plots/{config.name}/sources.png', dpi=400)

def plot_fluxes() -> None:
    """Plot the pseudomomentum flux time series for each strategy."""
    
    n_plots = len(COLORS) + 1
    n_cols = int(n_plots // 2)
    n_cols = n_cols + n_plots % 2
    n_rows = 2

    widths = [4.5] * n_cols + [0.3]
    fig = plt.figure(constrained_layout=True)
    fig.set_size_inches(sum(widths), 3 * n_rows)

    grid = gs.GridSpec(
        nrows=n_rows, ncols=(n_cols + 1),
        width_ratios=widths,
        figure=fig
    )

    axes = [fig.add_subplot(grid[i // 4, i % 4]) for i in range(n_plots)]
    cax = fig.add_subplot(grid[:, -1])

    strategies = ['reference'] + list(COLORS.keys())
    for i, (strategy, ax) in enumerate(zip(strategies, axes)):
        with open_dataset(f'data/{config.name}/{strategy}.nc') as ds:
            img, _ = plot_time_series(ds['pmf_u'] * 1000, amax=2, axes=[ax])
            ax.set_title(_format(strategy))

        if i < 4:
            ax.set_xlabel('')
        
        if i % 4 > 0:
            ax.set_ylabel('')

    cbar = plt.colorbar(img, cax=cax)
    cbar.set_ticks(np.linspace(-2, 2, 9))
    cbar.set_label('momentum flux (mPa)')

    plt.savefig(f'plots/{config.name}/fluxes.png', dpi=400)

def plot_scores() -> None:
    """Plot the error with respect to the reference for each strategy."""

    with _open_and_transform(f'data/{config.name}/reference.nc') as ds:
        ref = ds['pmf_u']
        z = ds['z'] / 1000

    fig, ax = plt.subplots()
    fig.set_size_inches(4, 6.5)

    for strategy, color in COLORS.items():
        with _open_and_transform(f'data/{config.name}/{strategy}.nc') as ds:
            pmf = ds['pmf_u']
            error = np.sqrt(((ref - pmf) ** 2).mean('time'))

            label = _format(strategy)
            ax.plot(error * 1000, z, color=color, label=label)

    label = 'average absolute PMF'
    scale = abs(ref).mean('time') * 1000
    ax.plot(scale, z, color='gray', ls='dashed', label=label)

    ax.set_xlim(0, 1.5)
    ax.set_ylim(z.min(), z.max())
    ax.grid(color='lightgray')

    xticks = np.linspace(0, 1.5, 4)
    ax.set_xticks(xticks)

    yticks = np.linspace(z.min(), z.max(), 7)
    ylabels = (10 * np.round((yticks / 10))).astype(int)
    ax.set_yticks(yticks, labels=ylabels)

    ax.set_xlabel('RMSE (mPa)')
    ax.set_ylabel('height (km)')
    ax.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(f'plots/{config.name}/scores.png', dpi=400)

def _format(strategy: str) -> str:
    """Format the name of a strategy to appear in plots."""

    if strategy == 'reference':
        return 'no-purge'

    return strategy.replace('e-', 'e, ')

def _open_and_transform(path: str) -> xr.Dataset:
    """Open a dataset, resample, and take only the time domain of interest."""

    ds = open_dataset(path)
    days = cftime.date2num(ds['time'], f'days since {EPOCH}')
    ds = ds.isel(time=((10 <= days) & (days <= 20)))
    ds = ds.resample(time='6h').mean('time')

    return ds