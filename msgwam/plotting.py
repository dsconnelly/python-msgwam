from __future__ import annotations
from typing import TYPE_CHECKING, Optional

import cftime
import matplotlib.gridspec as gs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from . import config
from .constants import EPOCH
from .sources import get_spectrum

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.collections import QuadMesh
    from matplotlib.colorbar import Colorbar

def plot_integration(ds: xr.Dataset, output_path: str) -> None:
    """
    Make a summary plot of an integration, including the mean wind and total,
    westerly, and easterly momentum flux time series.

    Parameters
    ----------
    ds
        Dataset containing the integration output.
    output_path
        Where to save the image.

    """

    widths = [4.5, 4.5, 0.2]
    fig = plt.figure(constrained_layout=True)
    fig.set_size_inches(sum(widths), 6)

    spec = gs.GridSpec(
        nrows=2, ncols=len(widths),
        width_ratios=widths,
        figure=fig
    )

    axes = [fig.add_subplot(spec[j // 2, j % 2]) for j in range(4)]
    caxes = [fig.add_subplot(spec[i, 2]) for i in range(2)]

    _, u_cbar = plot_time_series(ds['u'], 50, [axes[0], caxes[0]], 'PuOr_r')
    u_cbar.set_label('$\\bar{u}$ (m / s)') # type: ignore
    axes[0].set_title('mean zonal wind')

    names = ['total', 'westerly', 'easterly']
    pmfs = [ds['pmf_e'] + ds['pmf_w'], ds['pmf_e'], ds['pmf_w']]

    for name, pmf, ax in zip(names, pmfs, axes[1:]):
        _, cbar = plot_time_series(1000 * pmf, 2, [ax, caxes[1]])
        cbar.set_label('flux (mPa)') # type: ignore
        ax.set_title(f'{name} gravity wave flux')
        
    for ax in axes[:2]:
        ax.set_xlabel('')

    axes[1].set_ylabel('')
    axes[3].set_ylabel('')

    plt.savefig(output_path, dpi=400)

def plot_ray_count(ds: xr.Dataset, output_path: str) -> None:
    """
    Plot the number of active rays in an integration as a function of time.

    Parameters
    ----------
    ds
        Dataset containing the integration output.
    output_path
        Where to save the image.

    """

    fig, ax = plt.subplots()
    fig.set_size_inches(4.5, 3)

    time = cftime.date2num(ds['time'], f'days since {EPOCH}')
    ax.plot(time, ds['n_rays'], color='k')

    line = config.n_max * np.ones_like(time)
    ax.plot(time, line, color='gray', ls='dashed')

    tmax = time.max()
    ax.set_xlim(0, tmax)
    ax.set_xticks(np.linspace(0, tmax, 4))

    if config.n_increment == 0:
        ax.set_ylim(0, 1.1 * config.n_max)

    ax.set_xlabel('time (days)')
    ax.set_ylabel('active rays')

    ax.set_axisbelow(True)
    ax.grid(color='lightgray')
    ax.tick_params('both', direction='in')

    plt.tight_layout()
    plt.savefig(output_path, dpi=400)

def plot_source(ax: Optional[Axes]=None) -> Axes:
    """
    Make a bar plot of momentum flux as a function of phase speed for the source
    spectrum indicated by the loaded configuration settings.

    Parameters
    ----------
    ax
        Axis on which to plot. If `None`, a new axis will be created.

    Returns
    -------
    Axes
        Axis containing the source plot. If the `ax` parameter was not `None`,
        this is simply the same `Axes` as was provided.

    """

    if ax is None:
        fig, ax = plt.subplots()
        fig.set_size_inches(4.5, 3)

    ds = get_spectrum()
    cp_x = ds['cp_x'].values
    dc = abs(cp_x[1] - cp_x[0])
    
    flux = 1000 * ds['flux'].values
    ax.bar(cp_x, flux, width=dc, ec='k', fc='lightgray')

    xticks = np.linspace(-config.c_max, config.c_max, 5)
    ax.set_xlim(xticks.min(), xticks.max())
    ax.set_xticks(xticks)

    yticks = np.linspace(0, 0.3, 5)
    ax.set_ylim(yticks.min(), yticks.max())
    ax.set_yticks(yticks)

    total = abs(flux).sum()
    ax.set_title(f'total flux = {total:.2f} mPa')
    ax.set_xlabel('phase speed (m / s)')
    ax.set_ylabel('flux (mPa)')

    ax.set_axisbelow(True)
    ax.grid(color='lightgray')
    ax.tick_params('both', direction='in')

    return ax

def plot_time_series(
    data: xr.DataArray,
    amax: float,
    axes: Optional[list[Axes]]=None,
    cmap: str='RdBu_r'
) -> tuple[QuadMesh, Optional[Colorbar]]:
    """
    Plot data with time and height coordinates.

    Parameters
    ----------
    data
        Data to plot. Must have a `'time'` coordinate and a height coordinate
        starting with `'z_'`.
    amax
        Maximum absolute value to use in the symmetric norm.
    axes
        List containing the `Axes` object that should contain the mesh plot and,
        if a colorbar is to eb added, the `Axes` that will contain that as well.
        If `len(axes) == 1`, no colorbar will be created. If `axes` is not
        provided, then a new figure with two axes will be created.
    cmap
        Colormap to use in the mesh plot.

    Returns
    -------
    QuadMesh, Colorbar
        Result from `pcolormesh` and associated colorbar. If no colorbar axis
        was provided, then the second return value will instead be `None`.

    """

    if axes is None:
        widths = [4.5, 0.2]
        fig, axes = plt.subplots(ncols=2, width_ratios=widths)
        fig.set_size_inches(sum(widths), 3)

    time = cftime.date2num(data['time'], f'days since {EPOCH}')
    name = [s for s in data.coords if str(s).startswith('z_')][0]
    z = data[name].values / 1000

    img = axes[0].pcolormesh(
        time, z, data.T,
        vmin=-amax, vmax=amax,
        shading='nearest',
        cmap=cmap
    )

    tmax = time.max()
    axes[0].set_xlim(0, tmax)
    axes[0].set_xticks(np.linspace(0, tmax, 6))

    yticks = np.linspace(z.min(), z.max(), 7)
    ylabels = 10 * np.round((yticks - yticks.min()) / 10)
    ylabels = (ylabels + yticks.min()).astype(int)

    axes[0].set_ylim(z.min(), z.max())
    axes[0].set_yticks(yticks, labels=ylabels)

    axes[0].set_xlabel('time (days)')
    axes[0].set_ylabel('height (km)')

    try:
        cbar = plt.colorbar(img, cax=axes[1], orientation='vertical')
        cbar.set_ticks(np.linspace(-amax, amax, 5)) # type: ignore

    except IndexError:
        cbar = None

    return img, cbar
