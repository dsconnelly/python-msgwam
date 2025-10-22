from typing import Any, Optional

import numba as nb
import numpy as np
import xarray as xr

from ... import hyperparameters as hp

def get_bin_edges(n_bins: Optional[int]=None) -> np.ndarray:
    """
    Get the bin edges to use when projecting the ray volumes.

    Returns
    -------
    np.ndarray
        Array of `hp.n_bins + 1` bin edges. Note that the bins may be unequally
        spaced in phase speed space.

    """

    edges = np.linspace(0, 55, hp.generation.n_bins)
    edges = np.concatenate((edges, [100]))

    if n_bins is not None:
        left = edges[:-1].reshape(n_bins, -1)[:, 0]
        edges = np.concatenate((left, edges[-1:]))

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

    year, month, site, lat = get_info(n)
    path = f'data/ml-accel/context/{year}/{site}-{month}.nc'

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

def get_pdx(
    k: np.ndarray,
    l: np.ndarray,
    cp_hat: np.ndarray,
    n_bins: Optional[int]=None
) -> np.ndarray:
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

    edges = get_bin_edges(n_bins)
    cp_hat = np.clip(cp_hat, edges[0], edges[-1])
    out = np.argmax(cp_hat[:, None] <= edges[1:], axis=1)

    if n_bins is None:
        n_bins = hp.generation.n_bins

    quad = (k > 0) + 2 * (l > 0) + 3 * (k < 0) + 4 * (l < 0)
    out = (quad - 1) * n_bins + out
    out[np.isnan(cp_hat)] = -1

    return out.astype(np.int32)

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
            out[p, j] += frac * data[i]
