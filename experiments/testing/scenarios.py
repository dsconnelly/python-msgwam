import sys

import cftime
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

sys.path.insert(0, '.')
from msgwam import config
from msgwam.constants import EPOCH, RAD_EARTH
from msgwam.plotting import plot_time_series
from msgwam.utils import shapiro_filter

_REGIMES = {
    'tropics' : (0, 30),
    'midlatitudes' : (35, -140),
    'vortex' : (60, -140)
}

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

    _, cbar = plot_time_series(ds['u'], 50)
    cbar.set_label('$\\bar{u}$ (m / s)')

    plt.tight_layout()
    plt.savefig(f'plots/{config.name}/mean-state-{scenario}.png', dpi=400)
    ds.to_netcdf(f'data/{config.name}/mean-state-{scenario}.nc')

def _ICON() -> xr.Dataset:
    """
    Return a mean flow scenario taken from ICON data.

    Returns
    -------
    xr.Dataset
        Dataset containing the mean flow scenario.

    """

    regime = config.name.split('-')[-1]
    lat, lon = _REGIMES[regime]

    with xr.open_dataset('data/ICON/DJF-2324.nc') as ds:
        lats = np.rad2deg(ds['clat'].values)
        lons = np.rad2deg(ds['clon'].values)

        i = np.argmin((lats - lat) ** 2 + (lons - lon) ** 2)
        seconds = ds['time'].values.astype(int) / 1e9
        u = ds['u'].isel(ncells=i).values

    with xr.open_dataset('data/ICON/vgrid.nc') as ds:
        z = ds['z_ifc'].isel(ncells_2=i).values
        z = (z[:-1] + z[1:]) / 2

    time = cftime.num2date(seconds - seconds[0], f'seconds since {EPOCH}')
    faces = np.linspace(*config.grid_bounds, config.n_grid)
    centers = (faces[:-1] + faces[1:]) / 2

    ds = xr.Dataset({
        'time' : time,
        'z' : z,
        'u' : (('time', 'z'), u),
        'v' : (('time', 'z'), np.zeros_like(u))
    })

    kwargs = {'fill_value' : 'extrapolate'}
    ds = ds.interp(z=centers, kwargs=kwargs)

    return ds

def _jra_midlatitudes() -> xr.Dataset:
    """
    Return a mean flow scenario taken from JRA-55 in the midlatitudes.

    Returns
    -------
    xr.Dataset
        Dataset containing the mean flow scenario.

    """

    lat_slice = slice(-12, -14)
    lon_slice = slice(109, 111)

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

def _descending_jets() -> xr.Dataset:
    """
    Return a mean flow scenario with descending jets approximating the QBO.

    Returns
    -------
    xr.Dataset
        Dataset containing the mean flow scenario.

    """

    seconds = config.dt * np.arange(config.n_t_max)
    datetimes = cftime.num2date(seconds, f'seconds since {EPOCH}')
    faces = np.linspace(*config.grid_bounds, config.n_grid)
    centers = (faces[:-1] + faces[1:]) / 2

    k = 2 * np.pi / (14 * 86400)
    ell = 2 * np.pi / 25e3

    x, y = np.meshgrid(seconds, centers)
    wave = np.exp(1j * (k * x + ell * y)).real.T
    env = np.exp(-((centers - 45e3) / 10e3) ** 2)

    args = [config.n_t_max, config.n_grid - 1, 5 / 2]
    noise_1 = _make_colored_noise(*args)
    noise_2 = _make_colored_noise(*args)

    u = env * 40 * (wave + noise_1) + 5 * noise_2
    u[:, 1:-1] = shapiro_filter(u.T).T
    v = np.zeros_like(u)

    return xr.Dataset({
        'z' : centers,
        'time' : datetimes,
        'u' : (('time', 'z'), u),
        'v' : (('time', 'z'), v)
    })

def _make_colored_noise(n_t: int, n_z: int, p: float=3) -> np.ndarray:
    """
    Generate power law noise on a potentially unequal time-height grid.

    Parameters
    ----------
    n_t
        Number of points in the time dimension.
    n_z
        Number of points in the vertical dimension.
    p
        Power law governing noise.

    Returns
    -------
    np.ndarray
        Two-dimensional array of noise, ranging from -1 to 1.

    """

    ell = n_z * np.fft.fftfreq(n_z)
    k = n_t * np.fft.fftfreq(n_t)[:, None]
    wvn = np.sqrt((k ** 2 + ell ** 2) / (n_t ** 2 + n_z ** 2))

    A = np.zeros_like(wvn)
    A[wvn != 0] = 1 / wvn[wvn != 0]
    A = A ** (p / 2)

    phase = 2 * np.pi * np.random.rand(*A.shape)
    noise_hat = A * (np.cos(phase) * 1j * np.sin(phase))
    noise = np.fft.ifft2(noise_hat).real

    return 2 * (noise - noise.min()) / (noise.max() - noise.min()) - 1
