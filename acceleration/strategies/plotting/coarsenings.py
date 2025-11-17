import matplotlib.gridspec as gs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

from msgwam import config

from ...hyperparameters import scenarios as hp

from ..coarsenings import get_global_scores
from ..utils import by_kind, get_rnames

@by_kind
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

            zaxes[k].set_ylim(20, config.z_max / 1000)
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