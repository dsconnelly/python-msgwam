from __future__ import annotations
from os.path import dirname as parent
from typing import TYPE_CHECKING, Optional

import cftime
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from . import config, sources
from .constants import EPOCH
from .dispersion import cg_r, cp_x

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.collections import QuadMesh
    from matplotlib.colorbar import Colorbar

_REPO_DIR = parent(parent(__file__))
_font_path = f'{_REPO_DIR}/data/fonts/Lato-Regular.ttf'
_prop = fm.FontProperties(fname=_font_path)
fm.fontManager.addfont(_font_path)

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = _prop.get_name()
plt.rcParams['figure.dpi'] = 400

plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"

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

def plot_source(ax: Optional[Axes]=None) -> Axes:
    """
    Plot the source spectrum specified by the current config settings.

    Parameters
    ----------
    ax
        `Axes` object to plot with. If `None`, a new axis will be created and
        the size of the figure will be specified.

    Returns
    -------
    Axes
        `Axes` object used to plot. If `ax` is provided as an argument, the same
        object is returned.

    """

    if ax is None:
        fig, ax = plt.subplots()
        fig.set_size_inches(4.5, 3)

    cls_name = config.source_type.capitalize() + 'Source'
    source: sources.Source = getattr(sources, cls_name)()
    k, l, m, dk, dl, dm, dens = source.data[2:, 0]

    cp = cp_x(k, l, m)
    cg = cg_r(k, l, m)
    dc = abs(dm * cg / k)

    flux = k * cg * dens * abs(dk * dl * dm) * 1000
    ax.bar(cp, flux, dc, ec='k', fc='lightgray', zorder=2)

    ax.set_xlim(-config.c_max, config.c_max)
    ax.set_xticks(np.linspace(-config.c_max, config.c_max, 11))

    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks(np.linspace(-0.5, 0.5, 9))

    ax.set_xlabel('phase speed (m / s)')
    ax.set_ylabel('flux (mPa)')

    total = abs(flux).sum()
    ax.set_title(f'total flux: {total:.2f} mPa')
    ax.grid(color='lightgray', zorder=1)

    return ax

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
    time = cftime.date2num(data['time'], f'days since {EPOCH}')

    yticks = np.linspace(z.min(), z.max(), 7)
    ylabels = 10 * np.round((yticks - yticks.min()) / 10)
    ylabels = (ylabels + yticks.min()).astype(int)

    img = axes[0].pcolormesh(
        time, z, data.T,
        vmin=-amax, vmax=amax,
        shading='nearest',
        cmap='RdBu_r'
    )

    # levels = np.linspace(-amax, amax, 20)
    # img = axes[0].contourf(
    #     time, z, data.T,
    #     levels=levels,
    #     vmin=-amax, vmax=amax,
    #     cmap='RdBu_r'
    # )

    axes[0].set_xlabel('time (days)')
    axes[0].set_ylabel('height (km)')

    tmax = time.max()
    axes[0].set_xlim(0, tmax)
    axes[0].set_xticks(np.linspace(0, tmax, 4))

    axes[0].set_ylim(z.min(), z.max())
    axes[0].set_yticks(yticks, labels=ylabels)
    
    try:
        cbar = plt.colorbar(img, cax=axes[1], orientation='vertical')
        cbar.set_ticks(np.linspace(-amax, amax, 5)) # type: ignore

    except IndexError:
        cbar = None

    return img, cbar