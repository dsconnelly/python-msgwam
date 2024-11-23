from __future__ import annotations
from typing import TYPE_CHECKING, Optional
from warnings import catch_warnings

import cftime
import matplotlib.font_manager as fm
import matplotlib.gridspec as gs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

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

    ax.plot(days, pmf, color='k')
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

    _, u_cbar = plot_time_series(ds['u'], 50, [axes[0], caxes[0]], 'PuOr_r')
    u_cbar.set_label('$\\bar{u}$ (m / s)') # type: ignore
    axes[0].set_title('mean zonal wind')

    names = ['total', 'westerly', 'easterly']
    pmfs = [ds['pmf_e'] + ds['pmf_w'], ds['pmf_e'], ds['pmf_w']]

    for name, pmf, ax in zip(names, pmfs, axes[1:]):
        _, cbar = plot_time_series(1000 * pmf, 3, [ax, caxes[1]])
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

    source = ConstantSource()
    k, l, *_, flux = source._data.transpose(1, 0, 2)
    days = config.dt * np.arange(config.n_steps) / 86400
    cp_x = source._cp_x

    m = get_m(k, l, cp_x, config.N_ref)
    cg_r = get_cg_r(k, l, m, config.N_ref)

    widths = [4.5, 4.3, 0.2]
    fig = plt.figure(constrained_layout=True)
    fig.set_size_inches(1.25 * sum(widths), 1.25 * 3)

    spec = gs.GridSpec(
        nrows=3, ncols=3,
        width_ratios=widths,
        figure=fig
    )

    idxs = [(slice(None, None), 1), (0, 0), (1, 0), (2, 0)]
    axes = [fig.add_subplot(spec[*idx]) for idx in idxs]
    cax = fig.add_subplot(spec[:, 2])

    for ax in axes[1:]:
        ax.grid(color='lightgray')
        ax.set_axisbelow(True)

    vmax = 1000 * flux.max()
    vmax = np.ceil(vmax / 0.1) * 0.1

    img = axes[0].pcolormesh(
        days, cp_x, 1000 * flux.T,
        vmin=0, vmax=vmax,
        shading='nearest',
        cmap='Reds'
    )

    axes[0].set_xlim(0, config.n_day)
    axes[0].set_xticks(np.linspace(0, config.n_day, 6))
    axes[0].set_xlabel('time (days)')

    axes[0].set_ylim(-config.c_max, config.c_max)
    axes[0].set_yticks(np.linspace(*axes[0].get_ylim(), 5))
    axes[0].set_ylabel('$c_{\\mathrm{p}}$ (m / s)')

    cbar = plt.colorbar(img, cax=cax)
    cbar.set_label('flux (mPa)')
    
    names = ['$c_{\\mathrm{g}}$', '$\\lambda_x$', '$\\lambda_z$']
    fields = [cg_r, 2 * np.pi / abs(k) / 1000, 2 * np.pi / abs(m) / 1000]
    bounds = [(0, 5), (0, 1500), (0, 16)]
    units = ['m / s', 'km', 'km']

    zipped = zip(names, fields, bounds, units, axes[1:])
    for name, field, (a, b), unit, ax in zipped:
        edges = np.linspace(a, b, 13)
        x = (edges[:-1] + edges[1:]) / 2
        width = edges[1] - edges[0]

        h = 1e6 * _get_bin_averages(flux, field, edges)
        ax.bar(x, h, width=width, ec='k', fc='gray')

        ax.set_xlim(a, b)
        n_ticks = 5 if b == 16 else 6
        ax.set_xticks(np.linspace(a, b, n_ticks))
        ax.set_title(f'{name} ({unit})')

        ax.set_ylim(0, np.ceil(h.max() / 50) * 50)
        ax.set_yticks([*ax.get_ylim()])
        ax.set_ylabel('flux ($\mu$Pa)')

    axes[0].set_title('source flux')
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

    axes[0].set_xlim(time.min(), time.max())
    axes[0].set_xticks(np.linspace(time.min(), time.max(), 6))

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