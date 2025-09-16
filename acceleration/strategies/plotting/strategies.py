import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from matplotlib.colors import Normalize

from msgwam import config
from msgwam.sources import get_spectrum

from ...hyperparameters import scenarios as hp
from ...shared.constants import ACCEL_HOURS
from ...shared.plotting import plot_summaries

from ..overrides import get_overrides
from ..utils import load_data

def plot_spectrum(strategy: str, *args: str) -> None:
    """
    Plot the spectrum used by a particular strategy.

    Parameters
    ----------
    strategy, *args
        Strings to pass to `get_overrides`.

    """

    with config.override(**get_overrides(strategy, *args)):
        ds = get_spectrum()

    fig, ax = plt.subplots()
    fig.set_size_inches(4.5, 3)

    cp = ds['cp'].values
    flux = ds['flux'].values
    flux = len(flux) * flux / flux.sum()
    dc = ds['dc'].values

    ax.bar(
        cp, flux,
        width=dc,
        fc='lightgray',
        ec='k'
    )

    ax.set_xlim(0, config.c_max)
    ax.set_ylim(0, 1.5)

    s = 'c' if config.extrinsic else '\\hat{c}'
    ax.set_xlabel(f'${s}_{{\\mathrm{{p}}}}$ (m / s)')
    ax.set_ylabel('normalized source flux')
    ax.set_yticks([0, 0.5, 1, 1.5])

    plt.tight_layout()
    name = '-'.join([strategy] + list(args))
    kwargs = {'dpi' : 400, 'bbox_inches' : 'tight'}
    plt.savefig(f'plots/{config.name}/spectrum-{name}.png', **kwargs)

def plot_strategy(strategy: str, *args: str) -> None:
    """
    Plot a summary of the integration outputs, including the wind, momentum
    flux, and acceleration time series as well as the RMS profiles of each of
    those quantities. Other arguments specify additional behavior.

    Parameters
    ----------
    strategy
        Strategy for which to plot outputs.
    *args
        Optional arguments. If `'diff'` is included, the reference time series
        for each quantity will be subtracted off before plotting. Including
        `'components'` plots the four compass directions of momentum flux.

    """

    datas = {}
    amaxes = []
    units = []

    for c in hp.components:
        fields = [{'x' : 'u', 'y' : 'v'}[c]]
        labels = [f'$\\bar{{{fields[0]}}}$']
        factors = [1]

        amaxes = amaxes + [100]
        units = units + ['m / s']

        if 'components' in args:
            suffixes = {'x' : 'ew', 'y' : 'ns'}[c]
            fluxes = [f'pmf_{s}' for s in suffixes]
            labels = labels + [f'$F^{s.upper()}$' for s in suffixes]

        else:
            fluxes = [f'flux_{c}']
            labels = labels + [f'$F^{c}$']

        if abs(config.latitude) < config.lat_tropics:
            fmax = 5
        else:
            fmax = 10

        fields = fields + fluxes + [f'acceleration_{c}']
        factors = factors + [1e3] * len(fluxes) + [86400]
        labels = labels + [f'$D^{c}$']

        amaxes = amaxes + [fmax] * len(fluxes) + [100]
        units = units + ['mPa'] * len(fluxes) + ['m / s / d']
        
        for label, field, factor in zip(labels, fields, factors):
            kwargs = {}
            if field.startswith('acceleration'):
                kwargs['time_filter'] = ACCEL_HOURS * 3600

            use_diff = ('diff' in args) and field not in 'uv'
            load = lambda s: load_data(s, field, 0, **kwargs)

            ref = load('reference') if use_diff else 0
            datas[label] = factor * (load(strategy) - ref)

    plot_summaries(datas, amaxes, units)
    kwargs = dict(dpi=400, bbox_inches='tight')
    fname = '-'.join([strategy, *sorted(args)]) + '.png'
    plt.savefig(f'plots/{config.name}/{fname}', **kwargs)

def plot_trajectories(strategy: str, n_str: str='all') -> None:
    """
    Plot trajectories saved by `save-trajectories`.

    Parameters
    ----------
    strategy
        Strategy for which to plot the ray volume trajectories.

    """

    n_rows, n_cols = 2, 3
    fig, axes = plt.subplots(n_rows, n_cols)
    fig.set_size_inches(3 * n_cols, 4.5 * n_rows)
    axes = axes.flatten()

    cmap = cm.get_cmap('RdBu_r')
    norm = Normalize(-config.c_max, config.c_max)

    factors = [1 / 86400, 1 / 1000, 1, 1, 1000, 1000]
    names = ['age', 'dr', 'cp_hat', 'cg', 'energy', 'flux']
    bounds = [(0, 5), (0, 4), (-75, 75), (0, 3), (0, 50), (0, 5)]
    units = ['days', 'km', 'm / s', 'm / s', 'mJ / m$^3$', 'mPa']

    fname = f'{strategy}-trajectories.nc'
    with xr.open_dataset(f'data/{config.name}/strategies/{fname}') as ds:
        ds = ds.isel(meta=(ds['k'] != 0))

        if n_str != 'all':
            idx = np.argsort(np.random.rand(len(ds['meta'])))[:int(n_str)]
            ds = ds.isel(meta=idx)

        y = ds['r'].values / 1000
        for i in range(len(ds['meta'])):
            y = ds['r'].isel(meta=i).values / 1000
            color = cmap(norm(ds['cp_hat'].isel(meta=i, age=0)))

            for ax, name, factor in zip(axes, names, factors):            
                curve = factor * ds[name]
                if name != 'age': curve = curve.isel(meta=i)

                if name in ['energy', 'flux']:
                    curve = curve / ds['dr'].isel(age=0, meta=i)

                ax.plot(curve.values, y, color=color, alpha=0.05, lw=1)

    for ax, name, (xmin, xmax), unit in zip(axes, names, bounds, units):
        ax.set_xlabel(f'{name} ({unit})')
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(10, 60)

        ax.tick_params('both', direction='in')
        ax.grid(color='lightgray')

    plt.tight_layout()
    kwargs = dict(dpi=400, bbox_inches='tight')
    plt.savefig(f'plots/{config.name}/{strategy}-trajectories.png', **kwargs)
