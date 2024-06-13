import sys

import cftime
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

sys.path.insert(0, '.')
from msgwam import config
from msgwam.constants import EPOCH, RAD_EARTH
from msgwam.plotting import plot_time_series

def save_mean_state(scenario: str) -> None:
    """
    Save a mean state file for this config setup.

    Parameters
    ----------
    scenario
        Name of the background scenario. Must correspond to the name of one of
        the functions in scenarios.py.

    """

    func_name = '_' + scenario.replace('-', '_')
    ds: xr.Dataset = globals()[func_name]()

    _, cbar = plot_time_series(ds['u'], 45)
    cbar.set_label('$\\bar{u}$ (m s$^{-1}$)')

    plt.tight_layout()
    plt.savefig(f'plots/{config.name}/mean-state.png')
    ds.to_netcdf(f'data/{config.name}/mean-state.nc')

def _jra_midlatitudes() -> xr.Dataset:
    """
    Return a mean flow scenario taken from JRA-55 in the midlatitudes.

    Returns
    -------
    xr.Dataset
        Dataset containing the mean flow scenario.

    """

    lat_slice = slice(45, 35)
    lon_slice = slice(280, 290)

    kwargs = {'engine' : 'cfgrib', 'indexpath' : ''}
    with xr.open_dataset('data/JRA-55/gh.grib', **kwargs) as ds:
        ds = ds.sel(latitude=lat_slice, longitude=lon_slice)
        z = ds['gh'] * RAD_EARTH / (RAD_EARTH - ds['gh'])
        z = z.mean(('time', 'latitude', 'longitude'))
        z = z * config.grid_bounds[1] / z.max()
        
    with xr.open_dataset('data/JRA-55/u.grib', **kwargs) as ds:
        ds = ds.sel(latitude=lat_slice, longitude=lon_slice)
        u = ds['u'].mean(('latitude', 'longitude')).values
        v = np.zeros_like(u)

        seconds = (ds['time'] - ds['time'][0]).values.astype(int) / 1e9
        time = cftime.num2date(seconds, f'seconds since {EPOCH}')

    faces = np.linspace(*config.grid_bounds, config.n_grid)
    centers = (faces[:-1] + faces[1:]) / 2

    return xr.Dataset({
        'time' : time,
        'z' : z.values,
        'u' : (('time', 'z'), u),
        'v' : (('time', 'z'), v)
    }).interp(z=centers)

def _oscillating_jets() -> xr.Dataset:
    """
    Return a mean flow scenario with two oscillating jets.

    Returns
    -------
    xr.Dataset
        Dataset containing the mean flow scenario.

    """

    faces = np.linspace(*config.grid_bounds, config.n_grid)
    centers = (faces[:-1] + faces[1:]) / 2

    width = 3e3
    low = 10 * np.exp(-((centers - 25e3) / width) ** 2)
    high = 10 * np.exp(-((centers - 38e3) / width) ** 2)

    n_days = 35
    seconds = 86400 * np.linspace(0, n_days, 24 * n_days)
    datetimes = cftime.num2date(seconds, f'seconds since {EPOCH}')
    arg = 2 * np.pi * seconds / 86400 / 30

    shape = (len(seconds), len(centers))
    low = np.broadcast_to(low, shape) * np.sin(arg)[:, None]
    high = np.broadcast_to(high, shape) * np.cos(arg)[:, None]

    u = low - high
    v = np.zeros_like(u)

    return xr.Dataset({
        'z' : centers,
        'time' : datetimes,
        'u' : (('time', 'z'), u),
        'v' : (('time', 'z'), v)
    })

def _descending_jets() -> xr.Dataset:
    """
    Return a mean flow scenario with descending jets approximating the QBO.

    Returns
    -------
    xr.Dataset
        Dataset containing the mean flow scenario.

    """

    faces = np.linspace(*config.grid_bounds, config.n_grid)
    centers = (faces[:-1] + faces[1:]) / 2

    seconds = config.dt * np.arange(config.n_t_max)
    datetimes = cftime.num2date(seconds, f'seconds since {EPOCH}')

    k = 2 * np.pi / (10 * 86400)
    ell = 2 * np.pi / 25e3

    x, y = np.meshgrid(seconds, centers)
    env = 40 * np.exp(-((centers - 33e3) / 10e3) ** 2)
    u = env * np.exp(1j * (k * x + ell * y)).real.T
    v = np.zeros_like(u)

    return xr.Dataset({
        'z' : centers,
        'time' : datetimes,
        'u' : (('time', 'z'), u),
        'v' : (('time', 'z'), v)
    })