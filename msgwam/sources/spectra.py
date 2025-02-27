from math import prod
from typing import Any

import cftime
import numpy as np
import xarray as xr

from .. import config
from ..constants import EPOCH
from ..utils import cos_and_sin, get_time, make_colored_noise, open_dataset

def get_spectrum() -> xr.Dataset:
    """
    Return the properties of the spectrum specified by the loaded configuration
    file. Functions in this module (excluding utilities) should return a dataset
    with coordinates time (optional), phase speed, direction of propagation, and
    any other dimensions, and variables for dk, dl, intrinsic frequency, and
    flux. The dataset will be passed to `_postprocess`, and the remaining
    variables will be added at launch time by the `Source`.

    Returns
    -------
    xr.Dataset
        Dataset of wave properties.

    """

    func_name = '_' + config.spectrum_type
    return _postprocess(globals()[func_name]())

def _postprocess(ds: xr.Dataset) -> xr.Dataset:
    """
    Prepare a source dataset for use at the specific temporal and spectral
    resolutions set by the loaded configuration file.

    Parameters
    ----------
    ds
        Dataset as described in the docstring for `get_spectrum`.

    Returns
    -------
    xr.Dataset
        Dataset at the appropriate resolutions.

    """

    coords = set(ds.coords) - {'time'}
    sum_over = coords & set(ds['flux'].coords)
    total_flux = ds['flux'].sum(sum_over)

    n = prod([len(ds[c]) for c in coords - {'cp'}])
    ds = ds.interp(cp=_get_phase_velocities(config.n_source // n))
    ds['flux'] = total_flux * ds['flux'] / ds['flux'].sum(sum_over)

    if 'time' in ds.coords:
        time = get_time(config.dt_launch)
        ds = ds.sel(time=time, method='ffill')

    return ds.stack(channel=coords)[['dk', 'dl', 'omega_hat', 'flux']]

def _from_file() -> xr.Dataset:
    """Load a precomputed source spectrum from disk."""

    return open_dataset(config.spectrum_file)

def _desaubies() -> xr.Dataset:
    """Constant Desaubies background spectrum, as in Bölöni et. al (2021)."""

    phi = np.linspace(0, 2 * np.pi, config.n_phi + 1)[:-1]
    n = config.n_source // (config.n_phi * config.n_omega)
    cp = _get_phase_velocities(n)
    dphi = phi[1] - phi[0]

    bounds = [config.omega_hat_min, config.omega_hat_max]
    edges = np.linspace(*bounds, config.n_omega + 1)
    omega_hat = (edges[:-1] + edges[1:]) / 2

    m_star = 2 * np.pi / config.wvl_star
    top = cp * config.N_ref ** 3 * omega_hat[:, None] ** (1 - 5 / 3)
    bottom = config.N_ref ** 4 + m_star ** 4 * cp ** 4

    flux = m_star ** 3 * top / bottom
    flux = config.flux_bc * flux / flux.sum() / config.n_phi

    domega = np.diff(edges)[0]
    K = omega_hat[:, None] / cp
    p, q = domega / cp, K * dphi

    shape = (config.n_phi, config.n_omega, n)
    p = np.broadcast_to(p[None, None], shape)
    q = np.broadcast_to(q[None], shape)

    cos, sin = cos_and_sin(phi)
    cos = abs(cos)[:, None, None]
    sin = abs(sin)[:, None, None]

    dk = cos * p + sin * q
    dl = sin * p + cos * q

    copy = np.arange(config.n_omega)
    data = {'phi' : phi, 'copy' : copy, 'cp' : cp}

    data['dk'] = (('phi', 'copy', 'cp'), dk)
    data['dl'] = (('phi', 'copy', 'cp'), dl)
    data['omega_hat'] = ('copy', omega_hat)
    data['flux'] = (('copy', 'cp'), flux)

    return xr.Dataset(data)

def _gaussians() -> xr.Dataset:
    """
    Potentially variable-in-time source spectrum consisting of a Gaussian peak
    that may wander in phase speed space. The intrinsic frequency is constant in
    phase speed but may also evolve in time.
    """

    seconds = config.dt * np.arange(config.n_steps)
    decay_scale = 2 * np.pi * 86400 * config.tau_corr_days
    args = [seconds, decay_scale, 86400 * config.tau_cutoff_days]

    n_half = config.n_source // (2 * config.n_axes)
    cp = _get_phase_velocities(n_half)
    cp = np.concatenate((-cp, cp))

    flux = np.zeros((len(seconds), len(cp)))
    rng = np.random.default_rng(config.seed)

    for c_lo, c_hi in zip(config.c_los, config.c_his):
        center = make_colored_noise(
            *args,
            n_min=c_lo,
            n_max=c_hi,
            rng=rng
        )[:, None]

        arg = cp - center
        idx_in = np.sign(arg) != np.sign(center)

        arg[idx_in] = arg[idx_in] / config.c_width_in
        arg[~idx_in] = arg[~idx_in] / config.c_width_out
        flux = flux + np.exp(-0.5 * arg ** 2)

    cp = cp[n_half:]
    angle = np.deg2rad(config.direction)
    phi = np.array([angle + np.pi, angle])

    flux = config.flux_bc * flux / flux.sum(axis=1)[:, None] / config.n_axes
    flux = flux[:, None].reshape(-1, 2, n_half)

    if config.n_axes == 2:
        phi = np.concatenate((phi, phi + np.pi / 2))
        flux = np.hstack((flux, flux))

    bounds = [3600 * config.T_hat_lo, 3600 * config.T_hat_hi]
    omega_hat = 2 * np.pi / make_colored_noise(*args, *bounds, rng=rng)
    
    ones = np.ones_like(omega_hat)
    dk, dl = config.dk_init * ones, config.dl_init * ones
    stacked = np.stack((dk, dl, omega_hat), axis=0)

    time = cftime.num2date(seconds, f'seconds since {EPOCH}')
    data: dict[str, Any] = {'time' : time, 'phi' : phi, 'cp' : cp}
    for i, name in enumerate(['dk', 'dl', 'omega_hat']):
        data[name] = ('time', stacked[i])

    data['flux'] = (('time', 'phi', 'cp'), flux)
    ds = xr.Dataset(data)

    zipped = zip(config.c_los, config.c_his)
    varying = any([c_lo != c_hi for c_lo, c_hi in zipped])
    varying = varying or (config.T_hat_lo != config.T_hat_hi)

    if not varying:
        ds = ds.isel(time=0, drop=True)

    return ds

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

    bounds = np.linspace(0, config.c_max, n + 1)
    return (bounds[:-1] + bounds[1:]) / 2
