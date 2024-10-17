import sys

from typing import Iterator

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

PLOT_DIR = 'plots'
RESAMPLING = '3H'
SPINUP = 2

INCLUDE = {
    # 'inf' : ['hyperfine', 'fine', 'coarse'],
    'many' : ['fine', 'coarse'],
    'few' : ['fine', 'coarse']
    # 'few' : ['fine', 'coarse', 'stochastic-25', 'emulation', 'network']
    # 'few' : ['fine', 'coarse', 'intermittent', 'stochastic-1', 'stochastic-25']
    # 'few' : ['fine', 'coarse', 'stochastic-25', 'emulation', 'network']
}

COLORS = {
    'fine' : 'k',
    'coarse' : 'royalblue',
    'network' : 'tab:red',
    'intermittent' : 'darkviolet',
    'stochastic' : 'gold',
    'emulation' : 'forestgreen'
}

STYLES = {
    'fine' : 'dotted',
    'coarse' : 'dashed',
    'network' : 'dashed',
    'intermittent' : 'dotted',
    'stochastic' : 'dotted',
    'emulation' : 'dotted'
}

FACTORS = {'u' : 1, 'pmf_u' : 1000}
LONG_NAMES = {'u' : 'wind', 'pmf_u' : 'flux'}
UNITS = {'u' : 'm / s', 'pmf_u' : 'mPa'}

def plot_fluxes(mode: str) -> None:
    """
    Plot the pseudomomentum flux time series for each strategy.
    
    Parameters
    ----------
    mode
        Whether to plot data from `'prescribed'` or `'interactive'` runs.

    """

    strategies = list(_get_strategies())
    n_plots = len(strategies)

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

    for i, ((resources, resolution), ax) in enumerate(zip(strategies, axes)):
        path = f'data/{config.name}/{resources}-{resolution}-{mode}.nc'

        with open_dataset(path) as ds:
            img, _ = plot_time_series(ds['pmf_u'] * 1000, amax=2, axes=[ax])
            ax.set_title(_format(resources, resolution))

        if i < n_cols:
            ax.set_xlabel('')

        if i % n_cols > 0:
            ax.set_ylabel('')

    cax = fig.add_subplot(grid[:, -1])
    cbar = plt.colorbar(img, cax=cax)

    cbar.set_ticks(np.linspace(-2, 2, 9))
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

def plot_scores(mode: str, name: str) -> None:
    """
    Plot profiles analyzing the performance of various strategies.

    Parameters
    ----------
    mode
        Suffix indicating which files to open for analysis.
    name
        Whether to plot scores for `'pmf_u'` or `'u'`.

    """

    fig, axes = plt.subplots(ncols=len(INCLUDE), squeeze=False)
    fig.set_size_inches(3 * len(INCLUDE), 4.5)
    axes = axes[0]

    # with _open_partial(f'data/{config.name}/many-fine-{mode}.nc') as ds:
    with _open_partial(f'data/{config.name}/inf-hyperfine-{mode}.nc') as ds:
        ds = ds.resample(time=RESAMPLING).mean('time')

        ref = ds[name]
        z = ds['z'] / 1000
        rms = _get_rmse(ref, 0)

    strategies = list(_get_strategies())
    for resources, resolution in strategies:
        label = resolution
        ls = 'solid'

        if resolution.startswith('stochastic'):
            resolution, n_samples = resolution.split('-')
            n_samples = int(n_samples)

            label = f'stochastic, x{n_samples}'
            if n_samples == 1:
                ls = 'dashed'

        if resolution == 'emulation':
            label = 'NN emulator'

        if resolution == 'network':
            label = 'NN adjuster'

        path = f'data/{config.name}/{resources}-{resolution}-{mode}.nc'
        if resources == 'inf' and resolution == 'hyperfine':
            continue

        kwargs = {
            'color' : COLORS[resolution],
            'ls' : ls,
            # 'ls' : STYLES[resolution],
            'label' : label
        }

        with _open_partial(path) as ds:
            ds = ds.resample(time=RESAMPLING).mean('time')

            if resolution == 'stochastic':
                ds = ds.isel(sample=slice(None, n_samples)).mean('sample')

            rmse = _get_rmse(ref, ds[name])
            ax = axes[list(INCLUDE.keys()).index(resources)]
            ax.plot(FACTORS[name] * rmse, z, **kwargs)

        Nmax = {'many' : 2502, 'few' : 278}[resources]
        ax.set_title(f'$N_\\mathrm{{max}} = {Nmax}$')

    xmax = {'u' : 40, 'pmf_u' : 1}[name]
    xticks = np.linspace(0, xmax, 5)

    yticks = np.linspace(z.min(), z.max(), 7)
    ylabels = (10 * np.round((yticks / 10))).astype(int)
    
    for ax in axes:
        label = 'RMS flux'
        ax.plot(FACTORS[name] * rms, z, color='gray', ls='dashed', label=label)

        ax.set_xlim(0, xmax)
        ax.set_xticks(xticks)
        ax.set_xlabel(f'RMS {LONG_NAMES[name]} error ({UNITS[name]})')

        ax.set_ylim(z.min(), z.max())
        ax.set_yticks(yticks, labels=ylabels)
        ax.set_ylabel('height (km)')
        ax.grid(color='lightgray')

    axes[-1].legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/{config.name}/scores-{mode}-{name}.png', dpi=400)
        
def _format(resources: str, resolution: str) -> str:
    """
    Format the name of a strategy to appear in plots.
    
    Parameters
    ----------
    resources
        Resource specification.
    resolution
        Resolution specification.

    Returns
    -------
    str
        Formatted strategy name.

    """

    output = f'{resources}, {resolution}'
    if output == 'many, fine':
        output = output + ' ("ICON")'

    return output

def _get_strategies() -> Iterator[str]:
    """
    
    """

    for resources, resolutions in INCLUDE.items():
        for resolution in resolutions:
            yield resources, resolution
            
def _get_rmse(a: xr.DataArray, b: xr.DataArray) -> xr.DataArray:
    """
    Calculate the RMSE in time between two arrays.

    Parameters
    ----------
    a
        First array of values.
    b
        Second array of values. Alternatively, pass `b=0` to calculate the RMS
        values of `a` in time.

    Returns
    -------
    xr.DataArray
        RMS errors in time between `a` and `b`. Has the same coordinates as `a`
        and `b` except that `'time'` has been averaged out.

    """

    return np.sqrt(((a - b) ** 2).mean('time'))

def _open_partial(path: str) -> xr.Dataset:
    """
    Open a dataset, dropping the first `SPINUP` days.

    Parameters
    ----------
    path
        Location of the dataset on disk.

    Returns
    -------
    xr.Dataset
        Data excluding the first `SPINUP` days.
  
    """

    ds = open_dataset(path)
    days = cftime.date2num(ds['time'], f'days since {EPOCH}')
    keep = SPINUP <= days

    return ds.isel(time=keep)
