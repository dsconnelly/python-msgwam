from typing import Any

import numpy as np
import xarray as xr

from .. import config

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
    ds: xr.Dataset = globals()[func_name]()

    if config.spectrum_type != 'custom':
        ds = _coarsen(ds)

    return ds

def _coarsen(ds: xr.Dataset) -> xr.Dataset:
    """
    Regrid a dataset of source spectrum data to the phase velocity grid defined
    by the loaded configuration file, ensuring that total flux is conserved.

    Parameters
    ----------
    ds
        Dataset of wave properties, presumably on a phase velocity grid with
        more than `config.n_source` points.

    Returns
    -------
    xr.Dataset
        Coarsened dataset, unless the original dataset was already no finer than
        the configuration grid, in which case the original dataset is returned.

    """

    if len(ds['cp_x']) <= config.n_source:
        return ds
    
    flux = ds['flux']
    cp_x = _get_phase_velocities(config.n_source)
    ds = ds.interp(cp_x=cp_x, kwargs={'fill_value' : 'extrapolate'})

    bins = ds['cp_x'].sel(cp_x=flux['cp_x'], method='nearest')
    bins = bins.assign_coords(cp_x=flux['cp_x'])
    ds['flux'] = flux.groupby(bins).sum()

    return ds

def _gaussians() -> xr.Dataset:
    """
    Constant-in-time source spectrum consisting of two Gaussian peaks, symmetric
    about the origin in phase space. If `config.c_center` is zero, the two peaks
    coincide with one another.
    """

    phi = np.deg2rad(config.direction)
    wvn_hor = 2 * np.pi / config.wvl_hor
    k, l = wvn_hor * np.cos(phi), wvn_hor * np.sin(phi)
    cp_x = _get_phase_velocities(int(1e5))

    omega_hat_sq = cp_x ** 2 * k ** 2
    cp_x = cp_x[omega_hat_sq > config.f ** 2]

    k = k * np.sign(cp_x)
    dk = config.dk_init
    dl = config.dl_init

    shift = abs(cp_x) - config.c_center
    flux = np.exp(-0.5 * (shift / config.c_width) ** 2)
    flux = config.flux_bc * flux / flux.sum()

    ones = np.ones_like(cp_x)
    spectrum = np.vstack((k, l * ones, dk * ones, dl * ones, flux))

    data: dict[str, Any] = {'cp_x' : cp_x}
    for i, name in enumerate(['k', 'l', 'dk', 'dl', 'flux']):
        data[name] = ('cp_x', spectrum[i])

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
