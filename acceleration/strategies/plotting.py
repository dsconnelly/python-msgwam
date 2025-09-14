from collections import defaultdict
from typing import Callable

import matplotlib.cm as cm
import matplotlib.gridspec as gs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle

from msgwam import config
from msgwam.plotting import plot_time_series
from msgwam.sources import get_spectrum
from msgwam.utils import gaussian_filter, get_vertical_grids

from ..hyperparameters import scenarios as hp
from ..shared.constants import RMS_FILTERS, STRAT_FILTERS
from ..shared.plotting import plot_summaries

from .coarsenings import get_global_scores
from .overrides import get_overrides
from .utils import get_rmse, get_rnames, load_data

_COLORS = {
    'MiMAlike' : 'k',
    'ICONlike' : 'k',
    'instantaneous' : 'tab:red',
    'stochastic' : 'royalblue',
    'coarse' : 'forestgreen'
}

_STYLES = {
    'ICONlike' : 'dashed',

    'stochastic-25' : 'dashed',
    'stochastic-100' : 'dotted',

    'coarse-energy' : 'dashed',
    'coarse-cg_r' : 'dotted',
    'coarse-importance' : 'dashed'
}

_get_fields = lambda c: [f'flux_{c}', f'acceleration_{c}']
_get_labels = lambda c: [f'$F^{c}$', f'$D^{c}$']

def _by_kind(func: Callable) -> Callable:
    """
    Wrap a plotting function that takes a `kind` specifier so that if `'all'` is
    passed, all three `kind` options are called and plotted.
    """

    def wrapped(kind: str, *args, **kwargs):
        """Wrapped function to parse `kind` specifiers."""

        kinds = ['wind', 'flux', 'acceleration'] if kind == 'all' else [kind]
        for k in kinds: func(k, *args, **kwargs)

    return wrapped

@_by_kind
def plot_coarse_errors(kind: str, prefix: str='') -> None:
    """
    Plot the normalized error for each coarse resolution. If only plotting data
    from one run, also plot the RMSE as a function of height for each pair. If
    plotting data from multiple runs, show the average errors.

    Parameters
    ----------
    kind
        What field to plot the errors in.
    prefix
        Prefix to use to select runs to include.

    """

    rnames = get_rnames(prefix)

    if len(rnames) > 1:
        n_cols = len(hp.components)
        widths = [4.5] * n_cols + [0.2]

        fig = plt.figure(constrained_layout=True)
        spec = gs.GridSpec(1, n_cols + 1, fig, width_ratios=widths)
        fig.set_size_inches(sum(widths), 4.5)

        gaxes = [fig.add_subplot(spec[0, j]) for j in range(2)]
        caxes = [fig.add_subplot(spec[0, 2])]
        aaxes = gaxes

        scores = get_global_scores(kind, *rnames)

    else:
        n_rows = len(hp.components)
        widths = [3, 4.5, 0.2]

        fig = plt.figure(constrained_layout=True)
        spec = gs.GridSpec(n_rows, 3, fig, width_ratios=widths)
        fig.set_size_inches(sum(widths), n_rows * 4.5)

        zaxes = [fig.add_subplot(spec[i, 0]) for i in range(2)]
        gaxes = [fig.add_subplot(spec[i, 1]) for i in range(2)]
        caxes = [fig.add_subplot(spec[i, 2]) for i in range(2)]
        aaxes = [zaxes[0], gaxes[0], zaxes[1], gaxes[1]]

        path = f'data/{config.name}/coarsenings/coarse-errors-{kind}.nc'
        with xr.open_dataset(path) as ds:
            profiles = ds['error']
            rms = ds['rms']

        scores = (profiles / rms).mean('z', skipna=True)

    drs = scores['dr'].values
    n_sources = scores['n_source'].values
    dcs = [round(_get_dc(n // 2), 2) for n in n_sources]

    colors = ['darkgreen', 'w']
    cmap = LinearSegmentedColormap.from_list('custom', colors, 256)

    for k, c in enumerate(hp.components):
        grid = scores.sel(component=c)

        img = gaxes[k].imshow(
            grid.values.T,
            vmin=0, vmax=1,
            origin='lower',
            aspect='auto',
            cmap=cmap
        )

        gaxes[k].set_xticks(np.arange(len(drs)), labels=drs, rotation=45)
        gaxes[k].set_yticks(np.arange(len(n_sources)), labels=dcs)    

        gaxes[k].set_xlabel('$\\delta z$ (m)')
        gaxes[k].set_ylabel('$\\delta c_{\\mathrm{p}}$ (m / s)')

        if (len(rnames) == 1) or k == 0:
            cbar = plt.colorbar(img, cax=caxes[k], extend='max')
            cbar.set_ticks(np.linspace(0, 1, 5))
            cbar.set_label('normalized error')

        funcs = [grid.argmin, grid.argmax]
        colors = ['darkgreen', 'darkred']
        labels = ['best', 'worst']

        extrema = {}
        for func, color, label in zip(funcs, colors, labels):
            i, j = (da.item() for da in func(...).values())
            extrema[(i, j)] = (color, label, 1, 10)

            gaxes[k].add_patch(Rectangle(
                (i - 0.5, j - 0.5), 1, 1,
                ec=color, fc='none',
                clip_on=False,
                linewidth=2,
                zorder=10
            ))

        if len(rnames) == 1:
            defaults = ('k', None, 0.2, 1)
            curves = profiles.sel(component=c)
            z = curves['z'] / 1000

            factor, xmax, unit = {
                'wind' : (1, 50, 'm / s'),
                'flux' : (1000, 6, 'mPa'),
                'acceleration' : (86400, 100, 'm / s / day')
            }[kind]

            for i in range(curves.shape[0]):
                for j in range(curves.shape[1]):
                    color, label, alpha, zorder = extrema.get((i, j), defaults)
                    curve = factor * curves.isel(dr=i, n_source=j).values

                    zaxes[k].plot(
                        curve, z,
                        color=color,
                        label=label,
                        alpha=alpha,
                        zorder=zorder
                    )

            curve = factor * rms.sel(component=c).values
            zaxes[k].plot(curve, z, color='k', ls='dotted', label='RMS')

            xmin = 1e-1 if kind == 'acceleration' else 0
            zaxes[k].set_xlim(xmin, xmax)

            if kind == 'acceleration':
                zaxes[k].set_xscale('log')

            zaxes[k].set_ylim(10, config.z_max / 1000)
            zaxes[k].tick_params('both', direction='in')

            zaxes[k].grid(color='lightgray')
            zaxes[k].set_axisbelow(True)

            zaxes[k].set_xlabel(f'RMSE ({unit})')
            zaxes[k].set_ylabel('height (km)')

    if len(rnames) == 1:
        zaxes[1].legend()

    for i, ax in enumerate(aaxes):
        ax.set_title(f'({chr(i + 97)})')

    tag = '-global' if len(rnames) > 1 else ''
    path = f'plots/{config.name}/coarsenings-{kind}{tag}.png'
    plt.savefig(path, dpi=400, bbox_inches='tight')

def plot_components(strategy: str, mode: str='abs') -> None:
    """Plot individual (signed) components of the momentum flux."""

    widths = [4.5] * 3 + [0.2]
    fig, axes = plt.subplots(2, len(widths), width_ratios=widths)
    fig.set_size_inches(sum(widths), 6)

    for i, c in enumerate(hp.components):
        parts = {'x' : 'ew', 'y' : 'ns'}[c]
        fields = [f'flux_{c}'] + [f'pmf_{s}' for s in parts]
        amax = {'abs' : 3, 'diff' : 1}[mode]

        ref = 0
        kwargs = {
            'spinup_days' : 0,
            'time_filter' : 3 * 3600, 
            'z_filter' : 1000
        }

        for j, field in enumerate(fields):
            data = load_data(strategy, field, **kwargs)
            if mode == 'diff' : ref = load_data('reference', field, **kwargs)
            data = 1000 * (data - ref)

            img, _ = plot_time_series(data, amax, [axes[i, j]])
            axes[i, j].axhline(config.r_source / 1000, color='k', ls='dashed')
            axes[i, j].set_ylim(10, 60)

        cbar = plt.colorbar(img, axes[i, -1])
        cbar.set_ticks(np.linspace(-amax, amax, 5))
        cbar.set_label('mPa')

    plt.tight_layout()
    kwargs = {'dpi' : 400, 'bbox_inches' : 'tight'}
    path = f'plots/{config.name}/{strategy}-components-{mode}.png'
    plt.savefig(path, **kwargs)

def plot_ensemble_errors(strategy: str) -> None:
    """Plot errors as a function of ensemble size."""

    n_rows = len(hp.components)
    fig, ax = plt.subplots(n_rows, ncols=2)
    fig.set_size_inches(6, 4.5 * n_rows)

    if len(hp.components) == 2:
        axes = axes.T

    units = ['mPa', 'm / s / day']
    factors = [1000, 86400]    

    for i, c in enumerate(hp.components):
        zipped = zip(_get_fields(c), units, factors, _get_labels(c), axes[i])
        for j, (field, unit, factor, label, ax) in enumerate(zipped):
            z = get_vertical_grids()[j] / 1000
            ref = load_data('reference', field)
            z_filter = [None, 4e3][j]

            kwargs = {'z_filter' : z_filter, 'ensemble_mean' : False}
            data = load_data(strategy, field, **kwargs)

            cmap = plt.get_cmap('viridis_r')
            norm = Normalize(1, len(data['member']))

            for k in range(len(data['member'])):
                subset = data.isel(member=slice(None, k))
                rmse = factor * get_rmse(subset, ref)
                ax.plot(rmse, z, color=cmap(norm(k)))

            rms = factor * get_rmse(ref)
            ax.plot(rms, z, color='gray', ls='dashed', label='RMS')

            ax.set_xlim([0, 0][j], [3, 40][j])
            ax.set_ylim(config.z_min / 1e3, config.z_max / 1e3)

            ax.set_xlabel(f'{label} RMSE ({unit})')
            ax.set_ylabel('height (km)')

            ax.grid(color='lightgray')
            ax.tick_params('both', direction='in')

    plt.tight_layout()
    plt.savefig(f'plots/{config.name}/{strategy}-ensemble.png', dpi=400)

@_by_kind
def plot_error_profiles(kind: str, prefix: str, *strategies: str) -> None:
    """
    Plot RMS errors for various strategies, potentially averaged across runs.

    Parameters
    ----------
    kind
        What data to plot errors for. Must be either `'wind'`, `'flux'`, or
        `'acceleration'`, where `'wind'` only makes sense if the run is nudged
        or fully interactive.
    prefix
        The errors will be averaged over all runs in `data/` that share the
        given prefix. However, if `prefix` is the empty string, only this
        run will be considered.
    strategies
        Names of strategies to include in the plot.

    """

    rnames = get_rnames(prefix)
    oname = config.name

    tasks = ['abs']
    if len(rnames) > 1:
        tasks = tasks + ['rel']

    n_cols = len(hp.components)
    widths = [3.5] * n_cols + [0.75]
    fig, axes = plt.subplots(
        len(tasks), n_cols + 1,
        width_ratios=widths,
        squeeze=False
    )
    
    fig.set_size_inches(sum(widths), 4.5 * len(tasks))
    for ax in axes[:, -1]:
        ax.set_axis_off()

    for j, c in enumerate(hp.components):
        profiles = defaultdict(lambda: 0)
        ns = 0

        for rname in rnames:
            config.load(f'config/{rname}.toml')
            z = get_vertical_grids()[kind != 'flux'] / 1000

            drop = z * 1000 < config.r_source
            drop[-config.n_sponge:] = True
            ns = ns + (~drop).astype(int)

            if kind == 'wind':
                field = {'x' : 'u', 'y' : 'v'}[c]
            else:
                field = f'{kind}_{c}'
            
            ref = load_data(
                'reference',
                field=field,
                time_filter=None,
                z_filter=None
            )

            tmp = gaussian_filter(ref, **RMS_FILTERS)
            ref = gaussian_filter(ref, **STRAT_FILTERS)
            rms = get_rmse(tmp)

            for strategy in strategies:
                data = load_data(strategy, field)
                rmse = get_rmse(data, ref).values
                rmse[drop] = np.nan

                for task in tasks:
                    if task == 'rel':
                        rmse = np.minimum(1, rmse / rms)

                    tag = strategy + '-' + task
                    profiles[tag] = profiles[tag] + np.nan_to_num(rmse)

            if 'abs' in tasks:
                profiles['rms'] = profiles['rms'] + rms ** 2

        idx = ns > 0
        for i, task in enumerate(tasks):
            if kind == 'wind':
                factor, xmax = 1, 50
                field = {'x' : 'u', 'y' : 'v'}[c]
                cname = f'$\\bar{{{field}}}$'
                unit = 'm / s'

            elif kind == 'flux':
                factor, xmax = 1000, 5
                cname = f'$F_{{{c}}}$'
                unit = 'mPa'

            elif kind == 'acceleration':
                factor, xmax = 86400, 100
                cname = f'$D_{{{c}}}$'
                unit = 'm / s / day'

            if task == 'rel':
                factor = xmax = 1
                xlabel = f'{cname} normalized error'
            else:
                xlabel = f'{cname} RMSE ({unit})'

            handles = []
            for k, strategy in enumerate(strategies):
                tag = strategy + '-' + task
                curve = factor * profiles[tag]
                curve[idx] = curve[idx] / ns[idx]
                curve[~idx] = np.nan

                color = _COLORS[strategy.split('-')[0]]
                ls = _STYLES.get(strategy, 'solid')

                handles.append(axes[i, j].plot(
                    curve, z,
                    color=color, ls=ls,
                    label=_format_strategy(strategy),
                    zorder=(k + 2)
                )[0])

            if task == 'abs':
                rms = np.sqrt(profiles['rms'] / ns)

                handles.append(axes[i, j].plot(
                    factor * rms, z,
                    color='lightgray',
                    ls='dashed',
                    linewidth=1,
                    label='RMS',
                    zorder=-1
                )[0])

            if (task == 'abs') and (kind == 'acceleration'):
                xticks = 10. ** np.arange(-1, 3)
            else:
                xticks = np.linspace(0, xmax, 5)

            fmt = lambda v: str(int(v)) if int(v) == v else str(v)
            labels = list(map(fmt, xticks))

            axes[i, j].set_xlim(xticks.min(), xmax)
            axes[i, j].set_xticks(xticks, labels=labels)
            axes[i, j].set_ylim(20, config.z_max / 1e3)

            if kind == 'acceleration':
                axes[i, j].set_xscale('log')
                axes[i, j].minorticks_off()

            axes[i, j].set_xlabel(xlabel)
            axes[i, j].set_ylabel('height (km)')

            axes[i, j].grid(color='lightgray')
            axes[i, j].tick_params('both', direction='in')
            axes[i, j].set_axisbelow(True)

            if i + j == 0:
                axes[0, -1].legend(
                    handles=handles,
                    loc='lower left',
                    frameon=False
                )

    for i, ax in enumerate(axes[:, :-1].flatten()):
        ax.set_title(f'({chr(i + 97)})')

    plt.tight_layout()
    tag = '-global' if len(rnames) > 1 else ''
    path = f'plots/{oname}/errors-{kind}{tag}.png'
    plt.savefig(path, dpi=400, bbox_inches='tight')

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

    ax.set_xlabel('$c_{\\mathrm{p}}$ (m / s)')
    ax.set_ylabel('normalized source flux')
    ax.set_yticks([0, 0.5, 1, 1.5])

    plt.tight_layout()
    name = '-'.join([strategy] + list(args))
    kwargs = {'dpi' : 400, 'bbox_inches' : 'tight'}
    plt.savefig(f'plots/{config.name}/spectrum-{name}.png', **kwargs)

def plot_strategy(strategy: str) -> None:
    """
    Plot a summary of the integration outputs, including the momentum flux and
    acceleration time series as well as the RMS profiles of each quantity.

    Parameters
    ----------
    strategy
        Configuration strategy for which to plot the integration output.

    """

    datas = {}
    for c in hp.components:
        _iter = zip(*_get_plot_specs(c))
        extras = {l : x * load_data(strategy, f, 0) for l, f, x, *_ in _iter}
        datas.update(extras)

    *_, units, amaxes = _get_plot_specs(c)
    units = units * len(hp.components)
    amaxes = amaxes * len(hp.components)

    plot_summaries(datas, amaxes=amaxes, units=units)
    plt.savefig(f'plots/{config.name}/{strategy}.png', dpi=400)

def plot_trajectories(strategy: str) -> None:
    """
    Plot trajectories saved by `save-trajectories`.
    """

    n_rows, n_cols = 2, 3
    fig, axes = plt.subplots(n_rows, n_cols)
    fig.set_size_inches(3 * n_cols, 4.5 * n_rows)
    axes = axes.flatten()

    cmap = cm.get_cmap('RdBu_r')
    norm = Normalize(-config.c_max, config.c_max)

    factors = [1 / 86400, 1 / 1000, 1, 1, 1, 1000]
    names = ['age', 'dr', 'cp_hat', 'cg', 'action', 'flux']
    bounds = [(0, 5), (0, 3), (-75, 75), (0, 3), (0, 1.1), (0, 5)]

    fname = f'{strategy}-trajectories.nc'
    with xr.open_dataset(f'data/{config.name}/strategies/{fname}') as ds:
        ds = ds.isel(meta=(ds['k'] != 0))
        y = ds['r'].values / 1000

        for i in range(len(ds['meta'])):
            y = ds['r'].isel(meta=i).values / 1000
            color = cmap(norm(ds['cp_hat'].isel(meta=i, age=0)))

            for ax, name, factor in zip(axes, names, factors):            
                curve = factor * ds[name]
                if name != 'age': curve = curve.isel(meta=i)
                if name == 'action': curve = curve / curve[0]
                if name == 'flux': curve = curve / ds['dr'].isel(age=0, meta=i)

                ax.plot(curve.values, y, color=color, alpha=0.03, lw=1)

    for ax, name, (xmin, xmax) in zip(axes, names, bounds):
        ax.set_xlim(xmin, xmax)
        ax.set_xlabel(name)
        ax.set_ylim(10, 60)

        ax.tick_params('both', direction='in')
        ax.grid(color='lightgray')

    plt.tight_layout()
    kwargs = dict(dpi=400, bbox_inches='tight')
    plt.savefig(f'plots/{config.name}/{strategy}-trajectories.png', **kwargs)

def _format_strategy(strategy: str) -> str:
    """
    Format a strategy for display in a legend.

    Parameters
    ----------
    strategy
        Name of the strategy as it is saved to disk.

    Returns
    -------
    str
        More legible name for display.

    """

    if strategy.startswith('coarse'):
        _, suffix = strategy.split('-')
        return f'coarse ({suffix})'
    
    if strategy.startswith('stochastic'):
        _, n = strategy.split('-')
        return f'stochastic\n($\\epsilon = {n}^{{-1}}$)'
    
    if strategy == 'instantaneous':
        return 'steady-state'
    
    return strategy

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

def _get_plot_specs(
    c: str
) -> tuple[list[str], list[str], list[float], list[str], list[float]]:
    """
    Get information used to construct summary plots.

    Parameters
    ----------
    c
        Current component. Must be `'x'` or `'y'`.

    Returns
    -------
    list[str], list[str], list[float], list[str], list[float]
        Lists of labels, field names, scale factors, units, and axis maxima to
        use during plotting, respectively. Includes data for the mean wind if
        this scenario is interactive.

    """

    wind = {'x' : 'u', 'y' : 'v'}[c]
    labels = [f'$\\bar{{{wind}}}$', f'$F^{c}$', f'$D^{c}$']
    fields = [wind, f'flux_{c}', f'acceleration_{c}']

    factors = [1, 1e3, 86400]
    units = ['m / s', 'mPa', 'm / s / day']
    amaxes = [100, 3, 100]

    return labels, fields, factors, units, amaxes
