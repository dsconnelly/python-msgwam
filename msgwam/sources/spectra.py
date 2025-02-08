from typing import Any

import cftime
import numpy as np
import xarray as xr

from .. import config
from ..constants import EPOCH
from ..utils import get_time, make_colored_noise, open_dataset

def get_spectrum() -> xr.Dataset:
    """
    Return the properties of the spectrum specified by the loaded configuration
    file. Functions in this module (excluding utilities) should return a dataset
    with coordinates time (optional) and phase speed, and variables for the wave
    properties that can be determined without knowing the buoyancy frequency or
    the exact spectral resolution. These are omega_hat, phi, dk, dl, and flux.
    The other variables are added by `_postprocess` or by the Source object.

    Returns
    -------
    xr.Dataset
        Dataset of wave properties.

    """

    func_name = '_' + config.spectrum_type
    return _postprocess(globals()[func_name]())

def _postprocess(ds: xr.Dataset) -> xr.Dataset:
    """
    Prepare a source dataset for use at the specific resolution set by the
    loaded configuration file.

    Parameters
    ----------
    ds
        Dataset with coordinates `'cp_x'` and `'time'` along with variables
        `'omega_hat'`, `'phi'`, `'dk'`, `'dl'`, and `'flux'` defined along the
        time dimension only.

    Returns
    -------
    xr.Dataset
        Dataset at the right spectral and temporal resolution.

    """

    totals = ds['flux'].sum('cp_x')
    ds = ds.interp(cp_x=_get_phase_velocities(config.n_source))
    ds['flux'] = totals * ds['flux'] / ds['flux'].sum('cp_x')

    if 'time' in ds.coords:
        ds = ds.sel(time=get_time(config.dt_launch), method='ffill')

    return ds[['omega_hat', 'phi', 'dk', 'dl', 'flux']]

def _from_file() -> xr.Dataset:
    """Load a precomputed source spectrum from disk."""

    return open_dataset(config.spectrum_file)

def _gaussians() -> xr.Dataset:
    """
    Potentially variable-in-time source spectrum consisting of a Gaussian peak
    that may wander in phase speed space. The intrinsic frequency is constant in
    phase speed but may also evolve in time.
    """

    seconds = config.dt * np.arange(config.n_steps)
    decay_scale = 2 * np.pi * 86400 * config.tau_corr_days
    args = [seconds, decay_scale, 86400 * config.tau_cutoff_days]

    cp_x = _get_phase_velocities(config.n_source)
    flux = np.zeros((len(seconds), config.n_source))
    rng = np.random.default_rng(config.seed)

    for c_lo, c_hi in zip(config.c_los, config.c_his):
        center = make_colored_noise(
            *args,
            n_min=c_lo,
            n_max=c_hi,
            rng=rng
        )[:, None]

        flux = flux + np.exp(-0.5 * ((cp_x - center) / config.c_width) ** 2)

    flux = config.flux_bc * flux / flux.sum(axis=1)[:, None]

    omega_hat = 2 * np.pi / make_colored_noise(
        *args,
        n_min=(3600 * config.T_hat_lo),
        n_max=(3600 * config.T_hat_hi),
        rng=rng
    )

    ones = np.ones_like(omega_hat)
    phi = np.deg2rad(config.direction) * ones
    dk, dl = config.dk_init * ones, config.dl_init * ones
    stacked = np.stack((omega_hat, phi, dk, dl), axis=0)

    time = cftime.num2date(seconds, f'seconds since {EPOCH}')
    data: dict[str, Any] = {'time' : time, 'cp_x' : cp_x}
    data['flux'] = (('time', 'cp_x'), flux)

    for i, name in enumerate(['omega_hat', 'phi', 'dk', 'dl']):
        data[name] = ('time', stacked[i])

    return xr.Dataset(data)

def _get_phase_velocities(n: int) -> np.ndarray:
    """
    Return a grid of zonal phase velocities at source ray volume centers.

    Parameters
    ----------
    n
        How many points should be in the grid.

    Returns
    -------
    np.ndarray
        Zonal phase velocity at the center of each source ray volume.

    """

    bounds = np.linspace(-config.c_max, config.c_max, n + 1)
    return (bounds[:-1] + bounds[1:]) / 2
