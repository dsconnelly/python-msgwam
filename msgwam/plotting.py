from __future__ import annotations
from typing import TYPE_CHECKING, Optional

import cftime
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from matplotlib.cm import RdBu_r, ScalarMappable
from matplotlib.patches import Rectangle

from . import config, sources
from .constants import EPOCH
from .dispersion import cg_r, cp_x

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.collections import QuadMesh
    from matplotlib.colorbar import Colorbar

plt.rcParams['figure.dpi'] = 400

def plot_integration(ds: xr.Dataset, output_path: str) -> None:
    """
    Plot the mean zonal wind and pseudomomentum flux from an integration.

    Parameters
    ----------
    ds
        Open `xr.Dataset` containing integration output to plot.
    output_path
        Where to save the output image.

    """

    widths = [4.5, 0.2]
    fig, axes = plt.subplots(nrows=2, ncols=2, width_ratios=widths)
    fig.set_size_inches(sum(widths), 6)

    _, u_cbar = plot_time_series(ds['u'], 25, axes[0])
    _, pmf_cbar = plot_time_series(ds['pmf_u'] * 1000, 30, axes[1])

    u_cbar.set_label('$\\bar{u}$ (m s$^{-1}$)') # type: ignore
    pmf_cbar.set_label('PMF (mPa)') # type: ignore

    plt.tight_layout()
    plt.savefig(output_path)

def plot_source(amax: float, axes: Optional[list[Axes]]=None) -> ScalarMappable:
    """
    Plot the source spectrum specified by the current config settings.

    Parameters
    ----------
    amax
        Maximum absolute value to use in the symmetric norm.
    axes
        List containing the `Axes` object that should contain the color plot
        and, if a colorbar is to be added, the `Axes` that will contain the
        colorbar. If `len(axes) == 1`, no colorbar will be created. If `axes` is
        `None`, then a new figure will be created with two axes.

    Returns
    -------
    ScalarMappable
        Mappable object encoding the norm and colormap used.

    """

    if axes is None:
        widths = [4.5, 0.2]
        fig, axes = plt.subplots(ncols=2, width_ratios=widths)
        fig.set_size_inches(sum(widths), 3)

    cls_name = config.source_type.capitalize() + 'Source'
    source: sources.Source = getattr(sources, cls_name)()
    r, dr, k, l, m, dk, dl, dm, dens = source.data
    
    cp = cp_x(k, l, m)
    cg = cg_r(k, l, m)
    dc = abs(dm * cg / k)

    norm = plt.Normalize(vmin=-amax, vmax=amax)
    flux = k * cg * dens * abs(dk * dl * dm)
    colors = RdBu_r(norm(1000 * flux))

    for j, color in enumerate(colors):
        x = cp[j] - 0.5 * dc[j]
        y = r[j] - 0.5 * dr[j]

        axes[0].add_patch(Rectangle(
            (x, y), dc[j], dr[j],
            fc=color,
            ec='k'
        ))

    axes[0].set_xlim(-32, 32)
    axes[0].set_xticks(np.linspace(-32, 32, 9))
    axes[0].set_xlabel('phase speed (m / s)')

    axes[0].set_ylim(7.3e3, 9.8e3)
    ticks = np.linspace(7.3e3, 9.8e3, 6)
    axes[0].set_yticks(ticks, labels=(ticks / 1000))    
    axes[0].set_ylabel('height (km)')

    img = ScalarMappable(norm=norm, cmap=RdBu_r)

    if len(axes) > 1:
        plt.colorbar(img, cax=axes[1])

    return img

def plot_time_series(
    data: xr.DataArray,
    amax: float,
    axes: Optional[list[Axes]] = None
) -> tuple[QuadMesh, Optional[Colorbar]]:
    """
    Make a `pcolormesh` of data with time and height coordinates.

    Parameters
    ----------
    data
        Data to plot, along with `'z'` and `'time'` coordinates.
    amax
        Maximum absolute value to use in the symmetric norm.
    axes
        List containing the `Axes` object that should contain the color plot
        and, if a colorbar is to be added, the `Axes` that will contain the
        colorbar. If `len(axes) == 1`, no colorbar will be created. If `axes` is
        `None`, then a new figure will be created with two axes.

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

    z = data['z'].values / 1000
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
        cbar.set_ticks(np.linspace(-amax, amax, 5)) # type: ignore

    except IndexError:
        cbar = None

    return img, cbar