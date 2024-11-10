from __future__ import annotations
from typing import TYPE_CHECKING, Optional

import cftime
import matplotlib.gridspec as gs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from . import config
from .constants import EPOCH
from .dispersion import get_cg_r, get_m
from .sources import DeterministicSource

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.collections import QuadMesh
    from matplotlib.colorbar import Colorbar

def plot_boundary(ds: xr.Dataset, output_path: str) -> None:
    """
    Plot the flux at the bottom boundary over time.

    Parameters
    ----------
    Parameters
    ----------
    ds
        Dataset containing the integration output.
    output_path
        Where to save the image.

    """

    fig, ax = plt.subplots()
    fig.set_size_inches(4.5, 3)

    days = cftime.date2num(ds['time'], f'days since {EPOCH}')
    pmf = 1000 * (ds['pmf_e'] - ds['pmf_w']).isel(z_faces=0)
    line = 1000 * config.flux_bc * np.ones_like(days)

    ax.plot(days, pmf, color='k')
    ax.plot(days, line, color='gray', ls='dashed')

    ax.set_xlim(0, days.max())
    ax.set_ylim(0, 5)

    ax.set_xlabel('time (days)')
    ax.set_ylabel('boundary flux (mPa)')
    ax.set_title(f'mean flux = {pmf.mean():.2f} mPa')
    
    ax.set_axisbelow(True)
    ax.grid(color='lightgray')
    ax.tick_params('both', direction='in')

    plt.tight_layout()
    plt.savefig(output_path, dpi=400)

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
    amax = np.ceil(1000 * (pmfs[1].max() + 2 * pmfs[1].std()))

    for name, pmf, ax in zip(names, pmfs, axes[1:]):
        _, cbar = plot_time_series(1000 * pmf, amax, [ax, caxes[1]])
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

def plot_source(output_path: str) -> None:
    """
    Make Hovmöller plots of source momentum flux and vertical group velocity.

    Parameters
    ----------
    output_path
        Where to save the image.

    """

    source = DeterministicSource()
    k, l, *_, flux = source._data.transpose(1, 0, 2)
    days = config.dt * np.arange(config.n_steps) / 86400
    cp_x = source._cp_x

    m = get_m(k, l, cp_x, config.N_ref)
    cg_r = get_cg_r(k, l, m, config.N_ref)

    widths = [4.5, 0.2]
    fig, axes = plt.subplots(nrows=2, ncols=2, width_ratios=widths)
    fig.set_size_inches(sum(widths), 6)
    axes, caxes = axes.T

    vmax = 1000 * flux.max()
    vmax = np.ceil(vmax / 0.1) * 0.1

    img = axes[0].pcolormesh(
        days, cp_x,
        1000 * flux.T,
        vmin=0, vmax=vmax,
        shading='nearest',
        cmap='Reds'
    )

    cbar = plt.colorbar(img, cax=caxes[0])
    cbar.set_ticks(np.linspace(0, vmax, 5))
    cbar.set_label('flux (mPa)')

    vmax = 3.6 * cg_r.max()
    vmax = np.ceil(vmax / 5) * 5

    img = axes[1].pcolormesh(
        days, cp_x,
        3.6 * cg_r.T,
        vmin=0, vmax=vmax,
        shading='nearest',
        cmap='Blues'
    )

    cbar = plt.colorbar(img, cax=caxes[1])
    cbar.set_ticks(np.linspace(0, vmax, 5))
    cbar.set_label('$c_{\\mathrm{g}}$ (km / h)')

    for ax in axes:
        ax.set_xlim(0, days.max())

        ax.set_ylim(-config.c_max, config.c_max)
        ax.set_ylabel('$c_{\\mathrm{p}}$ (m / s)')
        ax.set_yticks(np.linspace(-config.c_max, config.c_max, 5))

    axes[0].set_xlabel('time (days)')

    plt.tight_layout()
    plt.savefig(output_path, dpi=400)

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
