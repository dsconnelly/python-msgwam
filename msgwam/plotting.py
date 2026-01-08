from __future__ import annotations
from typing import TYPE_CHECKING, Optional
from warnings import catch_warnings

import cftime
import matplotlib.font_manager as fm
import matplotlib.gridspec as gs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from matplotlib.colors import Normalize, SymLogNorm

from . import config
from .constants import EPOCH
from .dispersion import get_cg_r, get_m
from .sources import ConstantSource

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.collections import QuadMesh
    from matplotlib.colorbar import Colorbar

def init_plotting() -> None:
    """
    Set some aesthetically pleasing defaults for plots.
    """

    path = 'data/fonts/Lato-Regular.ttf'
    fm.fontManager.addfont(path)
    
    prop = fm.FontProperties(fname=path)
    plt.rcParams['font.sans-serif'] = prop.get_name()

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

    ax.plot(days, pmf.values.flatten(), color='k')
    ax.plot(days, line, color='gray', ls='dashed')

    ax.set_xlim(0, days.max())
    ax.set_ylim(0, 2000 * config.flux_bc)

    ax.set_xlabel('time (days)')
    ax.set_ylabel('boundary flux (mPa)')

    mean = pmf.isel(time=(days >= 1)).mean()
    ax.set_title(f'mean flux = {mean:.2f} mPa')
    
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

    _, u_cbar = plot_time_series(ds['u'], 75, [axes[0], caxes[0]], 'PuOr_r')
    u_cbar.set_label('$\\bar{u}$ (m / s)') # type: ignore
    axes[0].set_title('mean zonal wind')

    names = ['total', 'westerly', 'easterly']
    pmfs = [ds['pmf_e'] + ds['pmf_w'], ds['pmf_e'], ds['pmf_w']]

    for name, pmf, ax in zip(names, pmfs, axes[1:]):
        _, cbar = plot_time_series(1000 * pmf, 4, [ax, caxes[1]])
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
    Plot some informative properties of the source spectrum.

    Parameters
    ----------
    output_path
        Where to save the image.

    """

    widths = [4.5, 4.5, 4.5, 4.5, 0.2]
    fig = plt.figure(constrained_layout=True)
    fig.set_size_inches(sum(widths), 3)

    spec = gs.GridSpec(1, len(widths), fig, width_ratios=widths)
    axes = [fig.add_subplot(spec[0, i]) for i in range(5)]

    source = ConstantSource()
    k, l, *_, flux = source._data.transpose(1, 0, 2)
    days = config.dt * np.arange(config.n_steps) / 86400

    cp_x = source._cp_x
    m = get_m(k, l, cp_x, config.N_ref)
    cg_r = get_cg_r(k, l, m, config.N_ref)

    names = ['$\\lambda_x$', '$\\lambda_z$', '$c_{\\mathrm{g}}$']
    datas = [2 * np.pi / abs(k) / 1000, 2 * np.pi / abs(m) / 1000, cg_r]
    bounds = [(0, 2000), (0, 20), (0, 5)]
    units = ['km', 'km', 'm / s']
    
    zipped = zip(names, datas, bounds, units, axes[:3])
    for name, data, (a, b), unit, ax in zipped:
        edges = np.linspace(a, b, 13)
        x = (edges[:-1] + edges[1:]) / 2
        width = edges[1] - edges[0]

        h = 1e6 * _get_bin_averages(flux, data, edges)
        ax.bar(x, h, width=width, ec='k', fc='gray')

        ax.set_xlim(a, b)
        ax.set_ylim(0, 50)
        
        ax.grid(color='lightgray')
        ax.set_axisbelow(True)

        ax.set_xlabel(f'{name} ({unit})')
        ax.set_ylabel('flux ($\\mu$Pa)')

    vmax = 1000 * flux.max()
    vmax = np.ceil(vmax / 0.01) * 0.01

    img = axes[3].pcolormesh(
        days, cp_x, 1000 * flux.T,
        vmin=0, vmax=vmax,
        shading='nearest',
        cmap='Reds'
    )

    axes[3].set_xlabel('time (days)')
    axes[3].set_ylabel('$c_{\\mathrm{p}}$ (m / s)')

    cbar = plt.colorbar(img, cax=axes[4])
    cbar.set_label('source flux (mPa)')

    axes[0].set_title('(a) horizontal wavelength')
    axes[1].set_title('(b) vertical wavelength')
    axes[2].set_title('(c) vertical group velocity')
    axes[3].set_title('(d) source flux over time')

    plt.savefig(output_path, dpi=400)

def plot_time_series(
    data: xr.DataArray,
    amax: float,
    axes: Optional[list[Axes]]=None,
    cmap: str='RdBu_r',
    orientation: str='vertical',
    log_scale: bool=False
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
    orientation
        Orientation of colorbar, if one is created. Ignored otherwise.

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

    if log_scale:
        norm = SymLogNorm(1e-1, vmin=-amax, vmax=amax)
    else:
        norm = Normalize(vmin=-amax, vmax=amax)

    img = axes[0].pcolormesh(
        time, z, data.T,
        shading='nearest',
        norm=norm,
        cmap=cmap
    )

    axes[0].set_xlim(time.min(), time.max())
    axes[0].set_xticks(np.linspace(time.min(), time.max(), 6))

    yticks = np.linspace(z.min(), z.max(), 12)
    ylabels = 5 * np.round((yticks - yticks.min()) / 5)
    ylabels = (ylabels + yticks.min()).astype(int)

    axes[0].set_ylim(z.min(), z.max())
    axes[0].set_yticks(yticks, labels=ylabels)

    axes[0].set_xlabel('time (days)')
    axes[0].set_ylabel('height (km)')

    try:
        cbar = plt.colorbar(img, cax=axes[1], orientation=orientation)
        cbar.set_ticks(np.linspace(-amax, amax, 5)) # type: ignore

    except IndexError:
        cbar = None

    return img, cbar

def _get_bin_averages(
    data: np.ndarray,
    coord: np.ndarray,
    edges: np.ndarray
) -> np.ndarray:
    """
    Given a data array and an array giving a coordinate value for each data
    point, get the average data value for each of a series of coordinate bins.

    Parameters
    ----------
    data
        Data to average over.
    coord
        Coordinate values for each data point.
    edges
        Bin edges in coordinate space.

    Returns
    -------
    np.ndarray
        Array with `len(edges) - 1` elements of the average bin values.

    """

    out = np.zeros(len(edges) - 1)
    with catch_warnings(action='ignore', category=RuntimeWarning):
        for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            out[i] = data[(lo <= coord) & (coord < hi)].mean()

    return np.nan_to_num(out)