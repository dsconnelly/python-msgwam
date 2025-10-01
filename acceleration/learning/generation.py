from __future__ import annotations
from typing import TYPE_CHECKING, Any

import numba as nb
import numpy as np
import xarray as xr

from msgwam import config
from msgwam.integration import integrate
from msgwam.utils import get_vertical_grids

from ..hyperparameters import generation as hp

if TYPE_CHECKING:
    from msgwam.integration import _Callback
    from msgwam.means import MeanState
    from msgwam.propagators import TransientPropagator

def save_training_data() -> None:
    """Integrate and save the relevant quantities for training."""

    with config.override(**_get_overrides()):
        n_samples = 1 + (86400 * config.n_day) // config.dt_output
        seconds = np.arange(n_samples) * config.dt_output
        qnames = ['k > 0', 'l > 0', 'k < 0', 'l < 0']
        z_faces, z_centers = get_vertical_grids()

        wind = np.zeros((n_samples, 2, config.n_grid - 1))
        M, S = np.zeros((2, n_samples, 4 * hp.n_bins, config.n_grid - 1))
        D = np.zeros((n_samples, 4, config.n_grid - 1))
        F = np.zeros((n_samples, 4, config.n_grid))

        _ = integrate(_make_callback(wind, M, D, S, F))
        args = (n_samples, 4, hp.n_bins, config.n_grid - 1)
        M, S = M.reshape(*args), S.reshape(*args)

    data = {
        'time' : seconds.astype(int),
        'quadrant' : np.array(qnames),
        'bin' : np.arange(hp.n_bins),
        'z_centers' : z_centers,
        'z_faces' : z_faces
    }

    data['M_bulk'] = (('time', 'quadrant', 'bin', 'z_centers'), M)
    data['source'] = (('time', 'quadrant', 'bin', 'z_centers'), S)
    data['sink'] = (('time', 'quadrant', 'z_centers'), D)
    data['F_bulk'] = (('time', 'quadrant', 'z_faces'), F)
    
    for i, name in enumerate(['u', 'v']):
        data[name] = (('time', 'z_centers'), wind[:, i])

    xr.Dataset(data).to_netcdf(f'data/ml-accel/training/{config.name}.nc')

def _get_overrides() -> dict[str, Any]:
    """
    Return configuration overrides to while generating of training data.

    Returns
    -------
    dict[str, Any]
        Keyword arguments for `config.override`.

    """

    return {
        'n_max' : 5000,
        # 'dr_source' : -1800,
        # 'n_source' : 96,
        # 'dr_ghost' : 0,

        'dr_source' : 2000,
        'n_source' : 64,
        'n_day' : 30,

        'max_age' : -1,
        'min_flux' : 0,
        'prune_by' : 'none',
        'n_increment' : 1000,
        'strict_source' : True,

        'dt' : hp.dt,
        'dt_output' : hp.dt,
        'max_dt_multiplier' : 10,
    }

def _get_pdx(k: np.ndarray, l: np.ndarray, cp_hat: np.ndarray) -> np.ndarray:
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

    out = np.round(hp.n_bins * cp_hat / 100)
    quad = (k > 0) + 2 * (l > 0) + 3 * (k < 0) + 4 * (l < 0)
    out = (quad - 1) * hp.n_bins + np.clip(out, 0, hp.n_bins - 1)
    out[np.isnan(cp_hat)] = -1

    return out.astype(np.int32)

def _make_callback(
    wind: np.ndarray,
    M: np.ndarray,
    D: np.ndarray,
    S: np.ndarray,
    F: np.ndarray,
) -> _Callback:
    """
    Make a callback function to pass to the integrator.

    Parameters
    ----------
    B
        Array whose four dimensions range over time step, wavenumber quadrant,
        vertical grid_face, and property (bulk momentum and group velocity).
    S
        Array whose two dimensions range over time step and wavenumber quadrant,
        holding the momentum added to the system each time step.
    wind
        Array whose three dimensions range over time step, component of the
        mean wind, and vertical grid cell center.

    Returns
    -------
    _Callback
        Function that populates the provided arrays at each time step, given the
        current state of the system.
        
    """

    def callback(
        mean: MeanState,
        prop: TransientPropagator,
        n_step: int
    ) -> None:
        """Callback function to return as output."""

        cg = prop._get_cg_r(mean)
        wvn = abs((prop.k + prop.l))
        mom = wvn * prop.action

        cp_hat = prop._get_omega_hat(mean) / wvn
        bdx = _get_pdx(prop.k, prop.l, cp_hat)
        pdx = bdx // hp.n_bins
        ndx = bdx.copy()

        drop = prop.m > 0
        bdx[drop] = pdx[drop] = -1
        ndx[drop | (prop.age > 0)] = -1

        _project(prop.r, prop.dr, mean.z_faces, mom, bdx, M[n_step])
        _project(prop.r, prop.dr, mean.z_faces, mom, ndx, S[n_step])
        _project(prop.r, prop.dr, mean.z_faces, prop.attrition, pdx, D[n_step])
        _project(prop.r, prop.dr, prop._z_padded, mom * cg, pdx, F[n_step])

        wind[n_step] = mean.wind

    return callback

@nb.njit
def _project(
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
