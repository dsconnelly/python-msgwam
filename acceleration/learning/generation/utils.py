from typing import Any

import numba as nb
import numpy as np
import xarray as xr

from ... import hyperparameters as hp
from ...strategies import get_overrides as _get_strat_overrides

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

    year, month, site, lat = get_info(n)
    path = f'data/ml-accel/context/{year}/{site}-{month}.nc'
    kwargs = _get_strat_overrides('eulerian', 'fine')

    kwargs['dt'] = hp.generation.dt
    kwargs['dt_output'] = hp.generation.dt_output
    kwargs['prescribed_mean_file'] = path
    kwargs['latitude'] = lat

    return kwargs

def get_info(n: int) -> tuple[int, int, str, float]:
    """
    Given an integer (should be the SLURM task ID) get information relevant to
    setting up or integrating a particular month of training data.

    Parameters
    ----------
    int
        Current task ID.

    Returns
    -------
    int, int
        Year and month to pull MiMA data from.
    str
        Site name.
    float
        Latitude of the site.

    """

    n_site, month = divmod(n, 24)
    year, month = divmod(month, 12)
    month = month + 1
    year = 25 - year

    with xr.open_dataset('data/mima-scenarios-25.nc') as ds:
        site = ds['site'].values[n_site]
        lat = ds['lat'].values[n_site]

    return year, month, site, lat

def get_pdx(
    k: np.ndarray,
    l: np.ndarray,
    cpt: np.ndarray,
    edges: np.ndarray
) -> np.ndarray:
    """
    
    """

    cpt = np.clip(cpt, edges[0], edges[-1])
    out = np.argmax(cpt[:, None] <= edges[1:], axis=1)

    quad = (k > 0) + 2 * (l > 0) + 3 * (k < 0) + 4 * (l < 0)
    out = (quad - 1) * (len(edges) - 1) + out
    out[np.isnan(cpt)] = -1

    return out.astype(np.int32)

@nb.njit
def project(
    r: np.ndarray,
    dr: np.ndarray,
    edges: np.ndarray,
    data: np.ndarray,
    pdx: np.ndarray,
    out: np.ndarray,
) -> None:
    """
    JITted function that projects the momentum and group velocity contributions
    onto the vertical grid, and gets the indices of the most important rays for
    each grid level and wavenumber quadrant. Similar to the `project` function
    used by the MS-GWaM code proper, but specialized for use in the callback.
    """

    r_lo = r - 0.5 * dr
    r_hi = r + 0.5 * dr

    for i, (a, b, p) in enumerate(zip(r_lo, r_hi, pdx)):
        if np.isnan(a) or p < 0:
            continue

        for j, (z_lo, z_hi) in enumerate(zip(edges[:-1], edges[1:])):
            if b < z_lo:
                break

            if z_hi < a:
                continue

            frac = (min(b, z_hi) - max(a, z_lo)) / (z_hi - z_lo)

            for k in range(data.shape[0]):
                out[k, p, j] += frac * data[k, i]

