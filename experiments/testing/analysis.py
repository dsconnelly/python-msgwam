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

PLOT_DIR = 'plots'

COLORS = {
    'many-fine' : 'k',
    'few-fine' : 'gold',
    'many-coarse' : 'forestgreen',
    'few-coarse' : 'royalblue',
    'few-network' : 'tab:red',
    'intermittent' : 'darkviolet'
}

def plot_fluxes(mode: str) -> None:
    """
    Plot the pseudomomentum flux time series for each strategy.
    
    Parameters
    ----------
    mode
        Whether to plot data from `'prescribed'` or `'interactive'` runs.

    """

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

    axes = []
    for k in range(n_plots):
        i, j = k // n_cols, k % n_cols
        axes.append(fig.add_subplot(grid[i, j]))

    strategies = ['reference'] + list(COLORS.keys())
    for i, (strategy, ax) in enumerate(zip(strategies, axes)):
        path = f'data/{config.name}/{strategy}-{mode}.nc'

        with open_dataset(path) as ds:
            img, _ = plot_time_series(ds['pmf_u'] * 1000, amax=1, axes=[ax])
            ax.set_title(_format(strategy))

        if i < n_cols:
            ax.set_xlabel('')

        if i % n_cols > 0:
            ax.set_ylabel('')

    cax = fig.add_subplot(grid[:, -1])
    cbar = plt.colorbar(img, cax=cax)

    cbar.set_ticks(np.linspace(-1, 1, 9))
    cbar.set_label('momentum flux (mPa)')

    plt.savefig(f'{PLOT_DIR}/{config.name}/fluxes-{mode}.png', dpi=400)

def plot_lifetimes(mode: str) -> None:
    """
    Make box-and-whisker plots of the ray volume lifetimes in each integration.
    
    Parameters
    ----------
    mode
        Whether to plot data from `'prescribed'` or `'interactive'` runs.

    """

    fig, ax = plt.subplots()
    fig.set_size_inches(5, 4.5)

    names = ['reference'] + list(COLORS.keys())
    rates = np.zeros(len(names))
    lifetimes = []

    for k, name in enumerate(names):
        path = f'data/{config.name}/{name}-{mode}.nc'

        with open_dataset(path) as ds:
            launches = ds['age'].values == 0
            age = np.nan_to_num(ds['age'].values)
            idx = (age[1:] - age[:-1]) < 0

            rates[k] = launches.sum(axis=1).mean() / config.dt * 3600
            lifetimes.append(np.log(age[:-1][idx].flatten() / 3600))

    y = -np.arange(len(rates))
    ax.boxplot(
        lifetimes,
        positions=y,
        whis=(0, 100),
        vert=False,
        medianprops={'color' : 'k', 'ls' : 'dashed'}
    )

    labels = map(_format, names)
    ax.set_ylim(-len(rates) + 0.5, 0.5)
    ax.set_yticks(y, labels=labels)

    ticks = np.log(np.array([1 / 60, 1, 8, 24, 24 * 7]))
    labels = ['1 min', '1 hr', '8 hr', '1 day', '1 week']

    ax.set_xlim(ticks.min(), ticks.max())
    ax.set_xticks(ticks, labels=labels, rotation=30)
    ax.set_xlabel('ray volume lifetimes')

    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/{config.name}/lifetimes-{mode}.png', dpi=400)

def plot_scores(mode: str) -> None:
    """
    Plot the error with respect to the reference for each strategy.
    
    Parameters
    ----------
    mode
        Whether to plot data from `'prescribed'` or `'interactive'` runs. If
        interactive data is plotted, error will be plotted for mean wind too.

    """

    names = ['pmf_u']
    if mode == 'interactive':
        names = names + ['u']

    path = f'data/{config.name}/reference-{mode}.nc'
    with _open_and_transform(path) as ds:
        refs = {name: ds[name] for name in names}
        z = ds['z'] / 1000

    fig, axes = plt.subplots(ncols=len(names), squeeze=False)
    fig.set_size_inches(4 * len(names), 6.5)
    axes = axes[0]

    factors = {'u' : 1, 'pmf_u' : 1000}
    units = {'u' : 'm / s', 'pmf_u' : 'mPa'}
    long_names = {'u' : 'wind', 'pmf_u' : 'flux'}

    for (strategy, color), ax in zip(COLORS.items(), axes):
        path = f'data/{config.name}/{strategy}-{mode}.nc'
        label = _format(strategy)

        with _open_and_transform(path) as ds:
            for name in names:
                error = ds[name] - refs[name]
                profile = np.sqrt((error ** 2).mean('time'))
                ax.plot(profile * factors[name], z, color=color, label=label) 

    axes[0].legend(loc='upper right')
    for name, ax in zip(names, axes):
        label = f'RMS {long_names[name]}'
        scale = np.sqrt((refs[name] ** 2).mean('time')) * factors[name]
        ax.plot(scale, z, color='gray', ls='dashed', label=label)

        xmax = {'u' : 20, 'pmf_u' : 1.5}[name]

        ax.set_xlim(0, xmax)
        ax.set_ylim(z.min(), z.max())
        ax.grid(color='lightgray')

        xticks = np.linspace(0, xmax, 4)
        ax.set_xticks(xticks)

        yticks = np.linspace(z.min(), z.max(), 7)
        ylabels = (10 * np.round((yticks / 10))).astype(int)
        ax.set_yticks(yticks, labels=ylabels)

        ax.set_xlabel(f'RMSE ({units[name]})')
        ax.set_ylabel('height (km)')

    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/{config.name}/scores-{mode}.png', dpi=400)

def _format(strategy: str) -> str:
    """
    Format the name of a strategy to appear in plots.
    
    Parameters
    ----------
    strategy
        Name of the strategy being plotted.

    Returns
    -------
    str
        Formatted strategy name.

    """

    output = strategy.replace('-', ', ')
    if output == 'many, fine':
        output = output + ' ("ICON")'

    return output

def _open_and_transform(path: str) -> xr.Dataset:
    """
    Open a dataset, resample, and take only the time domain of interest.
    
    Parameters
    ----------
    path
        Path to netCDF dataset to be opened.

    Returns
    -------
    xr.Dataset
        Six-hourly averaged dataset, including only the specified time range.

    """

    ds = open_dataset(path)
    days = cftime.date2num(ds['time'], f'days since {EPOCH}')
    ds = ds.isel(time=((10 <= days) & (days <= 20)))
    ds = ds.resample(time='6h').mean('time')

    return ds