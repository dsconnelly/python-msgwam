from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from msgwam import config
from msgwam.utils import gaussian_filter, get_vertical_grids

from ...hyperparameters import scenarios as hp
from ...shared.constants import ACCEL_HOURS, RMS_FILTERS, STRAT_FILTERS

from ..utils import by_kind, get_rmse, get_rnames, load_data

_COLORS = {
    'MiMAlike' : 'k',
    'ICONlike' : 'k',
    'instantaneous' : 'tab:red',
    'stochastic' : 'royalblue',
    'coarse' : 'forestgreen',
    'network' : 'darkviolet'
}

_STYLES = {
    'ICONlike' : 'dashed',
    'stochastic-64' : 'dashed',
    'stochastic-100' : 'dotted'
}

@by_kind
def plot_error_profiles(kind: str, prefix: str, *strategies: str) -> None:
    """
    Plot RMS errors for various strategies, potentially averaged across runs.

    Parameters
    ----------
    kind
        What data to plot errors for. Must be either `'wind'`, `'flux'`, or
        `'acceleration'`. Note that `'wind'` will be useful only  if the run is
        nudged or fully interactive.
    prefix
        The errors will be averaged over all runs in `data/` that share the
        given prefix. However, if `prefix` is the empty string, only this
        run will be considered.
    strategies
        Names of strategies to include in the plot.

    """

    oname = config.name
    rnames = get_rnames(prefix)

    tasks = ['abs']
    if (len(rnames) > 1) or (kind == 'acceleration'):
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

            field = 'uv'['xy'.index(c)] if kind == 'wind' else f'{kind}_{c}'
            ref = load_data('reference', field, time_filter=None, z_filter=None)

            filters = STRAT_FILTERS.copy()
            if kind == 'acceleration':
                filters['hours'] = ACCEL_HOURS

            tmp = gaussian_filter(ref, **RMS_FILTERS)
            ref = gaussian_filter(ref, **filters)
            rms = get_rmse(tmp)

            for strategy in strategies:
                seconds = filters['hours'] * 3600
                data = load_data(strategy, field, time_filter=seconds)
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
            factor, xmax, cname, unit = _get_plot_specs(kind, c)
            factor, xmax = (1, 1) if task == 'rel' else (factor, xmax)
            xmax = 4 if len(rnames) > 1 and task == 'abs' else xmax

            suffix = f'normalized error' if task == 'rel' else f'RMSE ({unit})'
            xlabel = f'{cname} {suffix}'

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
                axes[i, j].set_xscale('log')
                axes[i, j].minorticks_off()

            else:
                xticks = np.linspace(0, xmax, 5)

            fmt = lambda v: str(int(v)) if int(v) == v else str(v)
            labels = list(map(fmt, xticks))

            axes[i, j].set_xlim(xticks.min(), xmax)
            axes[i, j].set_xticks(xticks, labels=labels)
            axes[i, j].set_ylim(20, config.z_max / 1e3)

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

    if strategy.startswith('MiMAlike'):
        return 'MiMA-like'

    if strategy.startswith('coarse'):
        _, *suffix = strategy.split('-')
        return f'optimal coarse\n(prune by {", ".join(suffix)})'
    
    if strategy.startswith('stochastic'):
        _, n = strategy.split('-')
        return f'stochastic\n($\\epsilon = {n}^{{-1}}$)'
    
    return strategy

def _get_plot_specs(kind: str, c: str) -> tuple[float, float, str, str]:
    """
    Get several useful plotting parameters depending on the field to plot.

    Parameters
    ----------
    kind
        Kind of data being plotted, as passed to `plot_error_profiles`.
    c
        Current component, either `'x'` or `'y'`.

    Returns
    -------
    factor
        Factor to scale profiles so that the units are reasonable.
    xmax
        Maximum value for the horizontal axis.
    cname
        Properly-rendered name to use in the horizontal axis label.
    unit
        Unit after scaling by `factor`.

    """

    if kind == 'wind':
        field = 'uv'['xy'.index(c)]
        return 1, 50, f'$\\bar{{{field}}}$', 'm / s'
    
    if kind == 'flux':
        xmax = 5 if abs(config.latitude) < config.lat_tropics else 10
        return 1000, xmax, f'$F^{{{c}}}$', 'mPa'
    
    if kind == 'acceleration':
        return 86400, 100, f'$D^{{{c}}}$', 'm / s / d'