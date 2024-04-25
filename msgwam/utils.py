import numpy as np
import xarray as xr

from . import config

def open_dataset(*args, **kwargs) -> xr.Dataset:
    """
    Open a netCDF file as an xarray Dataset. This function exists simply so that
    use_cftime=True can be the default.

    Parameters
    ----------
    args
        Positional arguments to xr.open_dataset.
    kwargs
        Keyword arguments to xr.open_dataset.

    Returns
    -------
    xr.Dataset
        The opened Dataset object, opened using cftime.

    """

    return xr.open_dataset(*args, use_cftime=True, **kwargs)

def _omega_hat(k: np.ndarray, l: np.ndarray, m: np.ndarray) -> np.ndarray:
    """
    Calculate the intrinsic frequency of internal gravity waves.

    Parameters
    ----------
    k
        Array of zonal wavenumbers.
    l
        Array of meridional wavenumbers.
    m
        Array of vertical wavenumbers.

    Returns
    -------
    np.ndarray
        Array of intrinsic frequencies.

    """

    return np.sqrt(
        (config.N0 ** 2 * (k ** 2 + l ** 2) + config.f0 ** 2 * m ** 2) /
        (k ** 2 + l ** 2 + m ** 2)
    )

def _cg_r(k: np.ndarray, l: np.ndarray, m: np.ndarray) -> np.ndarray:
    """
    Calculate the vertical group velocity of internal gravity waves.

    Parameters
    ----------
    k
        Array of zonal wavenumbers.
    l
        Array of meridional wavenumbers.
    m
        Array of vertical wavenumbers.

    Returns
    -------
    np.ndarray
        Array of vertical group velocities.

    """

    wvn_sq = k ** 2 + l ** 2 + m ** 2
    omega_hat = _omega_hat(k, l, m)

    return -m * (
        (omega_hat ** 2 - config.f0 ** 2) /
        (omega_hat * wvn_sq)
    )
