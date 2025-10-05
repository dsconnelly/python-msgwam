from typing import Any

import numpy as np
import xarray as xr

from ... import hyperparameters as hp

def get_bin_edges() -> np.ndarray:
    """
    Get the bin edges to use when projecting the ray volumes.

    Returns
    -------
    np.ndarray
        Array of `hp.n_bins + 1` bin edges. Note that the bins may be unequally
        spaced in phase speed space.

    """

    edges = np.linspace(0, 54, hp.generation.n_bins)
    edges = np.concatenate((edges, [100]))

    return edges

def get_overrides(n: int) -> dict[str, Any]:
    """
    Get the configurations to use while generating training data.

    Parameters
    ----------
    n
        Task ID for which to return configuration overrides.

    Returns
    -------
    dict[str, Any]
        Keyword arguments for `config.override`.

    """

    n_site = n // 12
    month = (n % 12) + 1
    site, lat = get_site_and_lat(n_site)
    path = f'data/ml-accel/context/{site}-{month}.nc'

    return {
        'prescribed_mean_file' : path,
        'latitude' : lat,

        'n_max' : 5000,
        'dr_source' : -hp.generation.dt_output,
        'n_source' : 128,
        'dr_ghost' : 0,
        
        'max_age' : 14 * 86400,
        'max_age_ghost' : 2 * 86400,
        'max_age_warning' : 86400,
        'min_flux' : 0,
        'min_cg' : 0,

        'prune_by' : 'none',
        'n_increment' : 1000,
        'strict_source' : True,
        'oob_action' : 'mark',

        'n_day' : 30,
        'dt' : hp.generation.dt,
        'dt_output' : hp.generation.dt_output,
        'max_dt_multiplier' : 10
    }

def get_pdx(k: np.ndarray, l: np.ndarray, cp_hat: np.ndarray) -> np.ndarray:
    """
    Return an integer array indicating the bin into which each ray should be
    projected. The rays are sorted by quadrant, and then perhaps more finely by
    intrinsic phase speed within each quadrant. 

    Parameters
    ----------
    k, l
        Arrays of zonal and meridional wavenumbers, respectively.
    cp_hat
        Absolute value of the intrinsic phase speed of each ray volume.

    Returns
    -------
    np.ndarray
        Index array giving the projection bin for each ray volume. Each quadrant
        gets `hp.n_bins` values before the next one. Inactive slots get -1.

    """

    edges = get_bin_edges()
    cp_hat = np.clip(cp_hat, edges[0], edges[-1])
    out = np.argmax(cp_hat[:, None] <= edges[1:], axis=1)

    quad = (k > 0) + 2 * (l > 0) + 3 * (k < 0) + 4 * (l < 0)
    out = (quad - 1) * hp.generation.n_bins + out
    out[np.isnan(cp_hat)] = -1

    return out.astype(np.int32)

def get_site_and_lat(n: int) -> tuple[str, float]:
    """
    Get the MiMA site name and latitude associated with a task ID.

    Parameters
    ----------
    n
        Current task ID.

    Returns
    -------
    str
        Associated MiMA site name.
    float
        Latitude of that site.

    """

    with xr.open_dataset('data/mima-scenarios.nc') as ds:
        site = ds['site'].values[n]
        lat = ds['lat'].values[n]

    return site, lat