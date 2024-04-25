from __future__ import annotations
from typing import TYPE_CHECKING, Optional

import cftime
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.collections import QuadMesh
    from matplotlib.colorbar import Colorbar

from .constants import EPOCH
from .mean import MeanFlow
from .rays import RayCollection
from .sources import _c_from, _m_from

plt.rcParams['figure.dpi'] = 400

def plot_integration(ds: xr.Dataset, output_path: str) -> None:
    """
    _summary_

    Parameters
    ----------
    ds
        Open xarray.Dataset with integration output to plot.
    output_path
        Where to save the output image.

    """

    widths = [4.5, 0.2]
    fig, axes = plt.subplots(nrows=2, ncols=2, width_ratios=widths)
    fig.set_size_inches(sum(widths), 6)

    _, u_cbar = plot_time_series(ds['u'], 25, axes[0])
    _, pmf_cbar = plot_time_series(ds['pmf_u'] * 1000, 30, axes[1])

    u_cbar.set_label('$\\bar{u}$ (m s$^{-1}$)')
    pmf_cbar.set_label('PMF (mPa)')

    plt.tight_layout()
    plt.savefig(output_path)

def plot_source(output_path: str, plot_cg: bool=True) -> None:
    """
    Plot the source spectrum defined in the loaded configuration file. Plots by
    zonal phase speed and, as such such, only plots waves with abs(k) > 0. Also
    can plot the vertical group velocity of each source wave.

    Parameters
    ----------
    output_path
        Where to save the output image.
    plot_cg
        Whether to also plot the vertical group velocities.

    """

    rays = RayCollection(MeanFlow())
    idx = abs(rays.sources[2]) > 0
    rays.data = rays.sources[:, idx]

    k = rays.k
    l = rays.l
    m = rays.m

    cg_r = rays.cg_r()
    data = 1000 * k * rays.action * cg_r
    cp = _c_from(k, l, m)
    
    n_cols = 1 + plot_cg
    fig, axes = plt.subplots(ncols=n_cols, squeeze=False)
    fig.set_size_inches(6 * n_cols, 4.5)

    axes = axes[0]
    axes[0].scatter(cp[cp < 0], data[cp < 0], color='k', zorder=5)
    axes[0].scatter(cp[cp > 0], data[cp > 0], color='k', zorder=5)
    axes[0].set_ylabel('momentum flux (mPa)')

    if plot_cg:
        axes[1].scatter(cp[cp < 0], cg_r[cp < 0], color='k', zorder=5)
        axes[1].scatter(cp[cp > 0], cg_r[cp > 0], color='k', zorder=5)
        axes[1].set_ylabel('group velocity (m / s)')

    ticks = np.linspace(-50, 50, 11)
    with np.errstate(divide='ignore'):
        ms = 1000 * _m_from(k[0], l[0], ticks)
        wvls = np.round(-2 * np.pi / ms, 2)

    for ax in axes:
        tx = ax.twiny()
       
        ax.set_xlim(ticks.min(), ticks.max())
        tx.set_xlim(ticks.min(), ticks.max())
        
        ax.set_xticks(ticks)
        tx.set_xticks(ticks)
        tx.set_xticklabels(wvls)
        
        ax.set_xlabel('phase speed (m / s)')
        tx.set_xlabel('$\\lambda_z$ (km)')

        ax.grid(True, color='lightgray', zorder=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=400)

def plot_time_series(
    data: xr.DataArray,
    amax: float,
    axes: Optional[list[Axes]]=None
) -> tuple[QuadMesh, Optional[Colorbar]]:
    """
    Plot data with time and height coordinates.

    Parameters
    ----------
    data
        DataArray containing the data along with 'z' and 'time' coordinates.
    axes
        List containing the Axes object that should contain the color plot and,
        if a colorbar is to be added, the Axes that will contain the colorbar.
        If len(axes) == 1, no colorbar will be created. If axes is None, then
        a new figure will be created with two axes.
    amax
        Maximum absolute value to use in the symmetric norm.

    Returns
    -------
    QuadMesh, Colorbar
        Result from pcolormesh and associated colorbar. If cax was None, then
        the second return value will be None as well.

    """

    if axes is None:
        widths = [4.5, 0.2]
        fig, axes = plt.subplots(ncols=2, width_ratios=widths)
        fig.set_size_inches(sum(widths), 3)
    
    z = data['z'] / 1000
    yticks = np.linspace(z.min(), z.max(), 7)
    ylabels = (10 * np.round((yticks / 10))).astype(int)
    time = cftime.date2num(data['time'], f'days since {EPOCH}')

    img = axes[0].pcolormesh(
        time, z, data.T,
        vmin=-amax, vmax=amax,
        shading='nearest',
        cmap='RdBu_r'
    )

    axes[0].set_xlabel('time (days)')
    axes[0].set_ylabel('height (km)')

    axes[0].set_xlim(0, time.max())
    axes[0].set_ylim(z.min(), z.max())
    axes[0].set_yticks(yticks, labels=ylabels)
    
    try:
        cbar = plt.colorbar(img, cax=axes[1])
        cbar.set_ticks(np.linspace(-amax, amax, 5))

    except IndexError:
        cbar = None

    return img, cbar