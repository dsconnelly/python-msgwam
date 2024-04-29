import sys

import cftime
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import numpy as np
import xarray as xr

sys.path.insert(0, '.')
from msgwam import config
from msgwam.constants import EPOCH
from msgwam.plotting import plot_time_series
from msgwam.utils import open_dataset

COLORS = {
    'ICON' : 'k',
    'do-nothing' : 'gold',
    'coarse-square' : 'forestgreen',
    'coarse-tall' : 'royalblue',
    'coarse-wide' : 'tab:red',
    'stochastic' : 'darkviolet'
}

# def plot_source() -> None:
#     _plot_source(f'plots/{config.name}/source.png')

def plot_fluxes() -> None:
    """Plot the pseudomomentum flux time series for each strategy."""
    
    n_plots = len(COLORS) + 1
    n_cols = int(n_plots // 2)
    n_cols = n_cols + n_plots % 2

    widths = [4.5] * n_cols + [0.3]
    fig = plt.figure(constrained_layout=True)
    fig.set_size_inches(sum(widths), 6)

    grid = gs.GridSpec(
        nrows=2, ncols=(n_cols + 1),
        width_ratios=widths,
        figure=fig
    )

    axes = [fig.add_subplot(grid[i // 4, i % 4]) for i in range(n_plots)]
    cax = fig.add_subplot(grid[:, -1])

    strategies = ['reference'] + list(COLORS.keys())
    for i, (strategy, ax) in enumerate(zip(strategies, axes)):
        with open_dataset(f'data/{config.name}/{strategy}.nc') as ds:
            img, _ = plot_time_series(ds['pmf_u'] * 1000, amax=1, axes=[ax])
            ax.set_title(_format(strategy))

        if i < 4:
            ax.set_xlabel('')
        
        if i % 4 > 0:
            ax.set_ylabel('')

    cbar = plt.colorbar(img, cax=cax)
    cbar.set_ticks(np.linspace(-1, 1, 9))
    cbar.set_label('momentum flux (mPa)')

    plt.savefig(f'plots/{config.name}/fluxes.png')

def plot_scores() -> None:
    """Plot the error with respect to the reference for each strategy."""

    with _open_and_transform(f'data/{config.name}/reference.nc') as ds:
        ref = ds['pmf_u']
        z = ds['z'] / 1000

    fig, ax = plt.subplots()
    fig.set_size_inches(4, 6.5)

    for strategy, color in COLORS.items():
        with _open_and_transform(f'data/{config.name}/{strategy}.nc') as ds:
            if strategy == 'ICON':
                runtime = ds.runtime

            pmf = ds['pmf_u']
            error = np.sqrt(((ref - pmf) ** 2).mean('time'))
            percent = ds.runtime / runtime * 100

            label = _format(strategy) + f' ({percent:.2f}%)'
            ax.plot(error * 1000, z, color=color, label=label)

    label = 'average absolute PMF'
    scale = abs(ref).mean('time') * 1000
    ax.plot(scale, z, color='gray', ls='dashed', label=label)

    ax.set_xlim(0, 0.8)
    ax.set_ylim(z.min(), z.max())
    ax.grid(color='lightgray')

    yticks = np.linspace(z.min(), z.max(), 7)
    ylabels = (10 * np.round((yticks / 10))).astype(int)
    ax.set_yticks(yticks, labels=ylabels)

    ax.set_xlabel('RMSE (mPa)')
    ax.set_ylabel('height (km)')
    ax.legend(loc='lower right')
    ax.set_title(config.name)

    plt.tight_layout()
    plt.savefig(f'plots/{config.name}/scores.png')

def _format(strategy: str) -> str:
    """Format the name of a strategy to appear in plots."""

    return strategy.replace('e-', 'e, ')

def _open_and_transform(path: str) -> xr.Dataset:
    """Open a dataset, resample, and take only the time domain of interest."""

    ds = open_dataset(path)
    days = cftime.date2num(ds['time'], f'days since {EPOCH}')
    ds = ds.isel(time=((10 <= days) & (days <= 20)))
    ds = ds.resample(time='6h').mean('time')

    return ds