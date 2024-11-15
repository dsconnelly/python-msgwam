from typing import Any

import cftime
import numpy as np
import xarray as xr

from .. import config
from ..constants import EPOCH
from ..utils import make_colored_noise

_N_LARGE = int(1e4)

def get_spectrum() -> xr.Dataset:
    """
    Return the properties of the spectrum specified by the loaded configuration
    file. Functions in this module (excluding utilities) should return a dataset
    with coordinates time (optional) and phase speed, and variables for the wave
    properties that can be determined without knowing the buoyancy frequency.
    These are k, l, dk, dl, and the momentum flux associated with each wave.

    Returns
    -------
    xr.Dataset
        Dataset of wave properties.

    """

    func_name = '_' + config.spectrum_type
    return globals()[func_name]()

def _coarsen(c_new: np.ndarray, c: np.ndarray, flux: np.ndarray) -> np.ndarray:
    """
    Coarsen an array of fluxes calculated on a very fine phase speed grid (for
    consistency across spectral resolutions) such that each flux on the coarse
    grid is the sum of all nearby fluxes on the fine grid.
    """

    p, = c_new.shape
    n_steps, q = flux.shape

    idx = np.repeat(np.arange(n_steps), q)
    jdx = np.argmin(abs(c_new - c[:, None]), axis=1)
    jdx = np.tile(jdx, n_steps)

    flux_new = np.zeros((n_steps, p))
    np.add.at(flux_new, (idx, jdx), flux.flatten())

    return flux_new

def _gaussians() -> xr.Dataset:
    """
    Potentially variable-in-time source spectrum consisting of a Gaussian peak
    that may wander in phase speed space. The intrinsic frequency is constant in
    phase speed but may also evolve in time.
    """

    seconds = config.dt * np.arange(config.n_steps)
    decay_scale = 2 * np.pi * 86400 * config.tau_corr_days
    args = [seconds, decay_scale, 3600 * config.tau_cutoff_hours]
    
    cp_fine = _get_phase_velocities(_N_LARGE)
    cp = _get_phase_velocities(config.n_source)
    flux = np.zeros((len(seconds), _N_LARGE))

    for c_lo, c_hi in zip(config.c_los, config.c_his):
        center = make_colored_noise(
            *args,
            n_min=c_lo,
            n_max=c_hi,
            seed=config.seed
        )[:, None]

        flux = flux + np.exp(-0.5 * ((cp_fine - center) / config.c_width) ** 2)

    flux = config.flux_bc * flux / flux.sum(axis=1)[:, None]
    flux = _coarsen(cp, cp_fine, flux)

    wvn_hor = 2 * np.pi / make_colored_noise(
        *args,
        n_min=(3600 * config.T_hat_lo),
        n_max=(3600 * config.T_hat_hi),
        seed=(config.seed + 1)
    )[:, None] / cp

    phi = np.deg2rad(config.direction)
    k, l = wvn_hor * np.cos(phi), wvn_hor * np.sin(phi)
    dk, dl = config.dk_init, config.dl_init

    ones = np.ones_like(k)
    spectrum = np.stack((k, l, dk * ones, dl * ones, flux), axis=0)

    time = cftime.num2date(seconds, f'seconds since {EPOCH}')
    data: dict[str, Any] = {'time' : time, 'cp_x' : cp}

    for i, name in enumerate(['k', 'l', 'dk', 'dl', 'flux']):
        data[name] = (('time', 'cp_x'), spectrum[i])

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
