import sys
import warnings

import cftime
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import xarray as xr

from cartopy.crs import PlateCarree

sys.path.insert(0, '.')
from msgwam import config
from msgwam.constants import EPOCH, RAD_EARTH

KWARGS = {
    'engine' : 'cfgrib',
    'backend_kwargs' : {'indexpath' : ''}
}

REGION = {
    'latitude' : slice(10, -10),
    'longitude' : slice(280, 290)
}

def plot_mean_state() -> None:
    widths = [4, 6.5, 0.2]
    fig = plt.figure(constrained_layout=True)
    fig.set_size_inches(sum(widths), 4)

    gs = gridspec.GridSpec(
        nrows=1, ncols=3,
        width_ratios=widths,
        figure=fig
    )

    lat_max, lat_min = REGION['latitude'].start, REGION['latitude'].stop
    lon_min, lon_max = REGION['longitude'].start, REGION['longitude'].stop
    
    ax = fig.add_subplot(gs[0, 0], projection=PlateCarree())
    ax.set_extent([lon_min, lon_max, lat_min, lat_max])

    with warnings.catch_warnings(action='ignore'):
        ax.imshow(
            plt.imread('data/raster.tif'),
            extent=[-180, 180, -90, 90],
            transform=PlateCarree()
        )
    
    xticks = np.linspace(lon_min, lon_max, 6) - 360
    yticks = np.linspace(lat_min, lat_max, 6)
    ax.set_xticks(xticks, crs=PlateCarree())
    ax.set_yticks(yticks, crs=PlateCarree())

    ax.set_xticklabels([f'{abs(int(x))}W' for x in xticks])
    ax.set_yticklabels([f'{int(x)}N' for x in yticks])
    
    ax.set_xlabel('longitude')
    ax.set_ylabel('latitude')

    ax = fig.add_subplot(gs[0, 1])
    with xr.open_dataset('data/baselines/mean-state.nc', use_cftime=True) as ds:
        days = cftime.date2num(ds['time'], units=f'days since {EPOCH}')
        z = ds['z'].values / 1000
        u = ds['u'].values

        vmax = 20
        levels = np.linspace(-vmax, vmax, 17)
        u = np.clip(u, -vmax, vmax)

        img = ax.pcolormesh(
            days, z, u.T,
            # levels=levels,
            shading='nearest',
            cmap='RdBu_r',
            vmin=-vmax, vmax=vmax,
            
        )

    ax.set_xlim(days.min(), days.max())
    ax.set_ylim(z.min(), z.max())

    yticks = np.linspace(z.min(), z.max(), 10)
    ylabels = np.linspace(*config.grid_bounds, 10) / 1000
    ax.set_yticks(yticks, labels=ylabels.astype(int))

    ax.set_xlabel('time (days)')
    ax.set_ylabel('height (km)')
    ax.set_title('prescribed zonal wind')

    cax = fig.add_subplot(gs[0, 2])
    cbar = plt.colorbar(img, cax=cax)
    cbar.set_label(r'$\bar{u}$ (m s$^{-1}$)')
    cbar.set_ticks(levels[::4])

    plt.savefig('plots/baselines/mean-state.png', dpi=400)

def save_mean_state() -> None:
    faces = np.linspace(*config.grid_bounds, config.n_grid)
    z = (faces[:-1] + faces[1:]) / 2

    width = 3e3
    low = 10 * np.exp(-((z - 24e3) / width) ** 2)
    mid = -10 * np.exp(-((z - 34e3) / width) ** 2)
    high = 10 * np.exp(-((z - 44e3) / width) ** 2)

    seconds = 86400 * np.linspace(0, 31, 128)
    datetimes = cftime.num2date(seconds, units=f'seconds since {EPOCH}')

    freq = 1 / (10 * 86400)
    envelope = np.sin(2 * np.pi * seconds * freq)
    u = np.broadcast_to(mid + high, (len(seconds), len(z)))
    u = u + envelope[:, None] * low

    xr.Dataset({
        'time' : datetimes,
        'z' : z,
        'u' : (('time', 'z'), u),
        'v' : (('time', 'z'), np.zeros_like(u))
    }).to_netcdf('data/baselines/mean-state.nc')

    return

    with xr.open_dataset('data/JRA-55/u.grib', **KWARGS) as ds:
        u = ds['u'].sel(**REGION).mean(('latitude', 'longitude'))

    with xr.open_dataset('data/JRA-55/gh.grib', **KWARGS) as ds:
        Phi = 9.8 * ds['gh'].sel(**REGION).mean(('latitude', 'longitude'))
        z = Phi * RAD_EARTH / (9.8 * RAD_EARTH - Phi)
        z = z * config.grid_bounds[1] / z.max()

    seconds = (u['time'].values - u['time'].values[0]).astype(int) / 1e9
    datetimes = cftime.num2date(seconds, units=f'seconds since {EPOCH}')
    faces = np.linspace(*config.grid_bounds, config.n_grid)
    centers = (faces[:-1] + faces[1:]) / 2

    xr.Dataset({
        'time' : datetimes,
        'z' : z.mean('time').values,
        'u' : (('time', 'z'), 0.5 * u.values),
        'v' : (('time', 'z'), np.zeros(u.shape))
    }).interp(z=centers).to_netcdf('data/baselines/mean-state.nc')
