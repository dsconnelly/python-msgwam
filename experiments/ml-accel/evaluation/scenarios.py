from typing import Optional

import cftime
import numpy as np
import xarray as xr

from msgwam import config
from msgwam.constants import EPOCH
from msgwam.sources.spectra import _gaussians
from msgwam.utils import get_vertical_grids, make_colored_noise, shapiro_filter

from ..hyperparameters import evaluation as hp

def save_descending_jets() -> None:
    """
    Save a mean wind field consisting of descending jets alternating westerly
    and easterly. The mean state is saved at the reference resolution, from
    which it can be coarsened later.
    """

    with config.override(dt=30, n_grid=501):
        path = f'data/{config.name}/input/descending-jets.nc'
        _get_descending_jets().to_netcdf(path)

def save_spectrum() -> None:
    """
    Save source spectrum data to disk, so that time is saved at the start of
    each integration and random perturbations are repeatable.
    """

    with config.override(spectrum_type='gaussians', n_source=int(1e3)):
        _gaussians().to_netcdf(config.spectrum_file)

def _get_descending_jets(
    n_day: Optional[int]=None,
    seed: int=7278
) -> xr.Dataset:
    """
    Generate a mean wind time series with descending zonal jets and zero
    meridional wind.

    Parameters
    ----------
    n_day
        How many days to generate a mean wind for. If `None`, the length of the
        integration specified by the configuration file will be used.
    seed
        Integer to use to seed the random number generator. Default value is the
        last four digit of somebody special's phone number.

    Returns
    -------
    xr.Dataset
        Dataset containing the mean wind time series.

    """

    if n_day is None:
        n_day = config.n_day

    n_steps = int(86400 * n_day / config.dt) + 1
    seconds = config.dt * np.arange(n_steps)

    units = f'seconds since {EPOCH}'
    time = cftime.num2date(seconds, units)
    _, z = get_vertical_grids()

    rng = np.random.default_rng(seed)
    args = [seconds, 5 * 86400, 5 * 86400]
    period_bounds = [hp.osc_period_min, hp.osc_period_max]
    period = make_colored_noise(*args, *period_bounds, rng) * 86400
    
    wvl = hp.wvl_max * np.ones_like(z)
    fade = (z - hp.z_turn) / (config.z_max - hp.z_turn)
    wvl[z > hp.z_turn] += (hp.wvl_min - hp.wvl_max) * fade[z > hp.z_turn]

    dz = z[1] - z[0]
    k = np.cumsum(1 / period) * config.dt
    ell = np.cumsum(1 / wvl)[:, None] * dz
    wave = np.exp(2j * np.pi * (k + ell)).real.T

    env = np.exp(-((z - hp.z_decay) / 25e3) ** 2)
    env_lo = np.exp(-((z - hp.z_decay) / 10e3) ** 2)
    env[z < hp.z_decay] = env_lo[z < hp.z_decay]

    k = seconds / (hp.jet_period * 86400)
    jet = np.exp(2j * np.pi * (k + rng.random())).real[:, None]
    args = [[seconds, z], [86400, 15e3], [3600, 500]]
    noise = make_colored_noise(*args, rng=rng)

    u = 10 * jet + 65 * env * wave + 15 * noise
    u[:, 1:-1] = shapiro_filter(u.T).T
    v = np.zeros_like(u)

    data = {'time' : time, 'z_centers' : z}
    data['u'] = (('time', 'z_centers'), u)
    data['v'] = (('time', 'z_centers'), v)

    return xr.Dataset(data)