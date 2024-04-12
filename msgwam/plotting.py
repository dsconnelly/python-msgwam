import cftime
import matplotlib.gridspec as gs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from .mean import MeanFlow
from .rays import RayCollection
from .sources import _c_from, _m_from

U_MAX = 25
PMF_MAX = 0.5

def plot_integration(ds: xr.Dataset, output_path: str) -> None:
    """
    Plot the zonal wind and pseudomomentum flux in integration output.

    Parameters
    ----------
    ds
        Open xarray.Dataset with data to plot.
    output_path
        Where to save the output image.
        
    """
    
    fig = plt.figure(constrained_layout=True)
    fig.set_size_inches(4.4, 6)

    grid = gs.GridSpec(nrows=2, ncols=2, figure=fig, width_ratios=[20, 1])
    axes = [fig.add_subplot(grid[i, 0]) for i in range(2)]
    caxes = [fig.add_subplot(grid[i, 1]) for i in range(2)]

    z = ds['z'] / 1000
    days = cftime.date2num(
        ds['time'],
        'days since 0000-01-01'
    )
    
    u = ds['u']
    pmf = ds['pmf_u'] * 1000

    u_plot = axes[0].pcolormesh(
        days, z, u.T,
        vmin=-U_MAX,
        vmax=U_MAX,
        cmap='RdBu_r',
        shading='nearest'
    )

    pmf_plot = axes[1].pcolormesh(
        days, z, pmf.T,
        vmin=-PMF_MAX,
        vmax=PMF_MAX,
        cmap='RdBu_r',
        shading='nearest'
    )

    u_bar = plt.colorbar(u_plot, cax=caxes[0])
    pmf_bar = plt.colorbar(pmf_plot, cax=caxes[1])

    axes[0].set_title('mean wind')
    axes[1].set_title('pseudomomentum flux')

    u_bar.set_label('$u$ (m s$^{-1}$)')
    pmf_bar.set_label('$c_\mathrm{g} k\mathcal{A}$ (mPa)')

    u_bar.set_ticks(np.linspace(-U_MAX, U_MAX, 5))
    pmf_bar.set_ticks(np.linspace(-PMF_MAX, PMF_MAX, 5))

    for ax in axes:
        ax.set_xlabel('time (days)')
        ax.set_ylabel('height (km)')

        ax.set_xlim(0, days.max())
        ax.set_ylim(z.min(), z.max())
        ax.set_yticks(np.linspace(z.min(), z.max(), 7))

    plt.savefig(output_path, dpi=400)

def plot_lifetimes(ds: xr.Dataset, output_path: str) -> None:
    """
    Plot a histogram of ray lifetimes in integration output.

    Parameters
    ----------
    ds
        Open xarray.Dataset with data to plot.
    output_path
        Where to save the output image.

    """

    age = np.nan_to_num(ds['age'].values)
    deltas = age[1:] - age[:-1]
    idx = deltas < 0

    data = -deltas[idx] / 3600
    y, edges = np.histogram(data, bins=48, range=(0, 96))
    x = (edges[:-1] + edges[1:]) / 2
    width = edges[1] - edges[0]

    fig, ax = plt.subplots()
    fig.set_size_inches(6, 4.5)
    ax.bar(x, y, width=width, color='lightgrey', edgecolor='k')
    
    ax.set_xlim(edges[0], edges[-1])
    ax.set_xlabel('lifetime (hr)')
    ax.set_ylabel('count')

    plt.tight_layout()
    plt.savefig(output_path, dpi=400)

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
    cp = np.sign(k) * _c_from(k, l, m)
    
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
