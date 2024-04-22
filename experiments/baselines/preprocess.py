import sys

import cftime
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

sys.path.insert(0, '.')
from msgwam import config
from msgwam.constants import EPOCH, RAD_EARTH

KWARGS = {
    'engine' : 'cfgrib',
    'backend_kwargs' : {'indexpath' : ''}
}

REGION = {
    'latitude' : slice(45, 35),
    'longitude' : slice(280, 290)
}

def plot_mean_state() -> None:
    widths = [6.5, 0.2]
    fig, (ax, cax) = plt.subplots(ncols=2, width_ratios=widths)
    fig.set_size_inches(sum(widths), 4)

    with xr.open_dataset('data/baselines/mean-state.nc', use_cftime=True) as ds:
        days = cftime.date2num(ds['time'], units=f'days since {EPOCH}')
        z = ds['z'].values / 1000
        u = ds['u'].values

        amax = 15
        img = ax.pcolormesh(
            days, z, u.T,
            shading='nearest',
            vmin=-amax, vmax=amax,
            cmap='RdBu_r'
        )

        exclude = (days < 10) | (20 < days)
        ax.fill_between(
            days, config.grid_bounds[1],
            where=~exclude,
            color='forestgreen', 
            alpha=0.1
        )

    ax.set_xlim(days.min(), days.max())
    ax.set_ylim(z.min(), z.max())

    yticks = np.linspace(z.min(), z.max(), 7)
    ylabels = np.linspace(*config.grid_bounds, 7) / 1000
    ax.set_yticks(yticks, labels=ylabels.astype(int))

    ax.set_xlabel('time (days)')
    ax.set_ylabel('height (km)')
    ax.set_title('prescribed zonal wind')

    cbar = plt.colorbar(img, cax=cax)
    cbar.set_label(r'$\bar{u}$ (m s$^{-1}$)')
    cbar.set_ticks(np.linspace(-amax, amax, 7))

    plt.tight_layout()
    plt.savefig('plots/baselines/mean-state.png', dpi=400)

def save_mean_state(mode: str) -> None:
    faces = np.linspace(*config.grid_bounds, config.n_grid)
    centers = (faces[:-1] + faces[1:]) / 2
    
    if mode == 'synthetic':
        width = 3e3
        low = 10 * np.exp(-((centers - 25e3) / width) ** 2)
        high = 10 * np.exp(-((centers - 38e3) / width) ** 2)

        n_days = 35
        seconds = 86400 * np.linspace(0, n_days, 24 * n_days)
        envelope = np.sin(2 * np.pi * seconds / 86400 / 30)

        shape = (len(seconds), len(centers))
        dipole = np.broadcast_to(low - high, shape)
        u = envelope[:, None] * dipole

    elif mode == 'jra':
        with xr.open_dataset('data/JRA-55/gh.grib', **KWARGS) as ds:
            Phi = 9.8 * ds['gh'].sel(**REGION).mean(('latitude', 'longitude'))
            z = Phi * RAD_EARTH / (9.8 * RAD_EARTH - Phi)
            z = z * config.grid_bounds[1] / z.max()
            z = z.mean('time').values

        with xr.open_dataset('data/JRA-55/u.grib', **KWARGS) as ds:
            deltas = ds['time'].values - ds['time'].values[0]
            seconds = deltas.astype(int) / 1e9

            u = ds['u'].sel(**REGION).mean(('latitude', 'longitude'))
            u = u.assign_coords(isobaricInhPa=z).rename(isobaricInhPa='z')
            u = u.interp(z=centers).values

    datetimes = cftime.num2date(seconds, units=f'seconds since {EPOCH}')

    xr.Dataset({
        'z' : centers,
        'time' : datetimes,
        'u' : (('time', 'z'), u),
        'v' : (('time', 'z'), np.zeros_like(u))
    }).to_netcdf('data/baselines/mean-state.nc')
