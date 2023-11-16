import sys

import matplotlib.gridspec as gs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

U_MAX = 20
PMF_MAX = 1

def make_plots(ds: xr.Dataset, output_path: str) -> None:
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

    u_bar = plt.colorbar(u_plot, cax=caxes[0])#, orientation='horizontal')
    pmf_bar = plt.colorbar(pmf_plot, cax=caxes[1])#, orientation='horizontal')

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

    plt.savefig(output_path, dpi=400)

if __name__ == '__main__':
    data_path, output_path = sys.argv[1:]
    with xr.open_dataset(data_path) as ds:
        make_plots(ds, output_path)
