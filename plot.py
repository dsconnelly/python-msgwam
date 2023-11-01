import sys

import matplotlib.pyplot as plt
import xarray as xr

def make_plots(ds: xr.Dataset, output_path: str) -> None:
    fig, axes = plt.subplots(ncols=2)
    fig.set_size_inches(8, 3)

    days = ds['time'] / (3600 * 24)
    grid = ds['grid'] / 1000

    u = ds['u']
    amax = abs(u).max()

    axes[0].pcolormesh(
        days,
        grid,
        u.T,
        vmin=-amax,
        vmax=amax,
        cmap='RdBu_r',
        shading='nearest'
    )

    pmf = ds['pmf_u'] * 1000
    amax = abs(pmf).max()

    axes[1].pcolormesh(
        days,
        grid,
        pmf.T,
        vmin=-amax,
        vmax=amax,
        cmap='RdBu_r',
        shading='nearest'
    )

    axes[0].set_title('$u$ (m s$^{-1}$)')
    axes[1].set_title('$c_\mathrm{g} k\mathcal{A}$ (mPa)')

    for ax in axes:
        ax.set_xlabel('time (days)')
        ax.set_ylabel('height (km)')

        ax.set_xlim(0, days.max())
        ax.set_ylim(grid.min(), grid.max())

    plt.tight_layout()
    plt.savefig(output_path, dpi=400)

if __name__ == '__main__':
    data_path, output_path = sys.argv[1:]
    with xr.open_dataset(data_path) as ds:
        make_plots(ds, output_path)
