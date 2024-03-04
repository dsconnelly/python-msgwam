import matplotlib.gridspec as gs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from .mean import MeanFlow
from .rays import RayCollection
from .sources import _c_from, _m_from

U_MAX = 20
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

    grid = gs.GridSpec(
        nrows=2,
        ncols=2,
        figure=fig,
        width_ratios=[20, 1]
    )
    
    axes = [fig.add_subplot(grid[i, 0]) for i in range(2)]
    caxes = [fig.add_subplot(grid[i, 1]) for i in range(2)]

    days = ds['time'] / (3600 * 24)
    grid = ds['grid'] / 1000

    u = ds['u']
    pmf = ds['pmf_u'] * 1000

    u_plot = axes[0].pcolormesh(
        days,
        grid,
        u.T,
        vmin=-U_MAX,
        vmax=U_MAX,
        cmap='RdBu_r',
        shading='nearest'
    )

    pmf_plot = axes[1].pcolormesh(
        days,
        grid,
        pmf.T,
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
        ax.set_ylim(grid.min(), grid.max())
        ax.set_yticks(np.linspace(*ax.get_ylim(), 7))

    plt.savefig(output_path, dpi=400)

def plot_source(output_path: str) -> None:
    """
    Plot the source spectrum defined in the loaded configuration file.

    Parameters
    ----------
    output_path
        Where to save the output image.

    """

    rays = RayCollection(MeanFlow())
    source = rays.sources
    rays.data = source

    k = rays.k
    l = rays.l
    m = rays.m

    cg_r = rays.cg_r()
    flux = 1000 * rays.k * rays.action * cg_r
    cp = np.sign(k) * _c_from(k, l, m)

    fig, axes = plt.subplots(ncols=2)
    fig.set_size_inches(12, 4.5)

    axes[0].plot(cp[cp < 0], flux[cp < 0], marker='o', color='k', zorder=5)
    axes[0].plot(cp[cp > 0], flux[cp > 0], marker='o', color='k', zorder=5)

    axes[1].plot(cp[cp < 0], cg_r[cp < 0], marker='o', color='k', zorder=5)
    axes[1].plot(cp[cp > 0], cg_r[cp > 0], marker='o', color='k', zorder=5)

    ticks = np.linspace(-50, 50, 11)
    with np.errstate(divide='ignore'):
        ms = np.round(1000 * _m_from(k[0], l[0], ticks), 2)

    for ax in axes:
        tx = ax.twiny()
       
        ax.set_xlim(ticks.min(), ticks.max())
        tx.set_xlim(ticks.min(), ticks.max())
        
        ax.set_xticks(ticks)
        tx.set_xticks(ticks)
        tx.set_xticklabels(ms)
        
        ax.set_xlabel('phase speed (m / s)')
        tx.set_xlabel('m (1 / km)')

        ax.grid(True, color='lightgray', zorder=0)

    axes[0].set_ylabel('momentum flux (mPa)')
    axes[1].set_ylabel('group velocity (m / s)')

    plt.tight_layout()
    plt.savefig(output_path, dpi=400)
