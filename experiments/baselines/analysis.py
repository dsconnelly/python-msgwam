import sys

from typing import Optional

import cftime
import matplotlib.gridspec as gs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from matplotlib.animation import FuncAnimation

sys.path.insert(0, '.')
from msgwam import config
from msgwam.constants import EPOCH
from msgwam.mean import MeanFlow
from msgwam.rays import RayCollection
from msgwam.sources import _c_from

COLORS = {
    'ICON' : 'k',
    'do-nothing' : 'gold',
    'coarse-square' : 'forestgreen',
    'coarse-tall' : 'royalblue',
    'coarse-wide' : 'tab:red',
    'stochastic' : 'darkviolet'
}

def plot_fluxes() -> None:
    widths = [6.5] * 4 + [0.4]
    fig = plt.figure(constrained_layout=True)
    fig.set_size_inches(0.5 * sum(widths), 4)

    grid = gs.GridSpec(nrows=2, ncols=5, figure=fig, width_ratios=widths)
    axes = [fig.add_subplot(grid[i // 4, i % 4]) for i in range(7)]
    cax = fig.add_subplot(grid[:, -1])

    names = ['reference'] + list(COLORS.keys())
    for name, ax in zip(names, axes):
        with _open_dataset(f'data/baselines/{name}.nc', resample=None) as ds:
            days = cftime.date2num(ds['time'], f'days since {EPOCH}')
            kms = ds['z'] / 1000

            pmf = ds['pmf_u'].values
            img = ax.pcolormesh(
                days, kms,
                pmf.T * 1000,
                cmap='RdBu_r',
                vmin=-1, vmax=1,
                shading='nearest'
            )

        ax.set_xlim(days.min(), days.max())
        ax.set_ylim(kms.min(), kms.max())

        ticks = np.linspace(kms.min(), kms.max(), 4)
        labels = (np.linspace(*config.grid_bounds, 4) / 1000).astype(int)
        ax.set_yticks(ticks, labels=labels)

        ax.set_xlabel('time (days)')
        ax.set_ylabel('height (km)')
        ax.set_title(_format_name(name))

    cbar = plt.colorbar(img, cax=cax)
    cbar.set_ticks(np.linspace(-1, 1, 9))
    cbar.set_label('momentum flux (mPa)')

    plt.savefig('plots/baselines/fluxes.png', dpi=400)

def plot_power() -> None:
    widths = [2, 6.5, 0.2]
    fig, axes = plt.subplots(ncols=3, width_ratios=widths)
    fig.set_size_inches(sum(widths), 4)

    mean = MeanFlow()
    rays = RayCollection(mean)

    bin_width = 1
    n_bins = int(2 * 40 / bin_width)
    edges = np.linspace(-40, 40, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2

    ds_all = _open_dataset('data/baselines/reference.nc', resample='1D')
    kms = ds_all['z'].values / 1000

    line, = axes[0].plot(np.zeros_like(kms), kms, color='k')
    img = axes[1].pcolormesh(
        centers, kms,
        np.zeros((len(kms), n_bins)),
        vmin=0, vmax=10,
        shading='nearest',
        cmap='Greys'
    )
    
    def update(k):
        ds = ds_all.isel(time=k)
        data = [ds[prop].values for prop in rays.props]
        rays.data = np.vstack(data)

        u = ds['u'].values
        c = _c_from(ds['k'], ds['l'], ds['m']).values
        c = c + np.interp(rays.r, mean.r_centers, u)

        count = (~np.isnan(rays.action)).astype(int)
        by_level = count * mean.get_fracs(rays, mean.r_faces)
        heatmap = np.zeros((n_bins, len(kms)))

        for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            jdx = (lo <= c) & (c < hi)
            if not jdx.sum(): continue
            heatmap[i] = by_level[:, jdx].sum(axis=1)

        line.set_xdata(u)
        img.set_array(heatmap.ravel())

    axes[0].set_xlim(-12, 12)
    axes[0].set_xlabel('$\\bar{u}$ (m / s)')
    axes[1].set_xlabel('phase speed (m / s)')
    axes[0].set_ylabel('height (km)')
    
    for ax in axes[:-1]:
        ax.set_ylim(kms.min(), kms.max())
        ticks = np.linspace(kms.min(), kms.max(), 7)
        labels = (10 * np.round((ticks / 10))).astype(int)
        ax.set_yticks(ticks, labels=labels)

        ax.grid(color='lightgray')

    axes[0].set_title('$\\bar{u}$')
    axes[1].set_title('number of ray volumes')
    plt.tight_layout()

    FuncAnimation(
        fig=fig,
        func=update,
        frames = np.arange(len(ds_all['time'])),
        interval=20
    ).save('plots/baselines/power.mp4')

def plot_scores() -> None:
    with _open_dataset('data/baselines/reference.nc') as ds:
        z = ds['z'] / 1000
        ref = ds['pmf_u'] * 1000

    fig, ax = plt.subplots()
    fig.set_size_inches(4, 6.5)

    for strategy, color in COLORS.items():
        with _open_dataset(f'data/baselines/{strategy}.nc') as ds:
            pmf = ds['pmf_u'] * 1000
            error = np.sqrt(((ref - pmf) ** 2).mean('time'))

            label = _format_name(strategy)
            ax.plot(error, z, color=color, label=label)

    ax.set_xlim(0, 0.8)
    ax.set_ylim(z.min(), z.max())
    ax.grid(color='lightgray')

    ticks = np.linspace(z.min(), z.max(), 7)
    labels = (np.linspace(*config.grid_bounds, 7) / 1000).astype(int)
    ax.set_yticks(ticks, labels=labels)

    ax.set_xlabel('RMSE (mPa)')
    ax.set_ylabel('height (km)')
    ax.legend()

    plt.tight_layout()
    plt.savefig('plots/baselines/scores.png', dpi=400)

def _format_name(name: str) -> str:
    return name.replace('e-', 'e, ')

def _open_dataset(path, resample: Optional[str]='6H') -> xr.Dataset:
    ds = xr.open_dataset(path, use_cftime=True)
    days = cftime.date2num(ds['time'], units=f'days since {EPOCH}')
    ds = ds.isel(time=((10 <= days) & (days <= 20)))

    if resample is not None:
        ds = ds.resample(time=resample).mean('time')

    return ds
