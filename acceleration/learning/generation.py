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

        B = np.zeros((n_samples, 2, 4, config.n_grid))
        wind = np.zeros((n_samples, 2, config.n_grid - 1))
        S = np.zeros((n_samples, 4))

        _ = integrate(_make_callback(B, S, wind))
        
    data = {
        'time' : seconds.astype(int),
        'quadrant' : np.array(qnames),
        'z_centers' : z_centers,
        'z_faces' : z_faces
    }

    data['source'] = (('time', 'quadrant'), S)
    for i, name in enumerate(['M_bulk', 'F_bulk']):
        data[name] = (('time', 'quadrant', 'z_faces'), B[:, i])

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
        'dr_source' : -1200,
        'n_source' : 128,
        'dr_ghost' : 0,

        'prune_by' : 'none',
        'n_increment' : 1000,

        'dt' : hp.dt_fine,
        'dt_output' : hp.dt_coarse,
        'max_dt_multiplier' : 10,
    }

def _get_quadrant(k: np.ndarray, l: np.ndarray) -> np.ndarray:
    """
    Return an integer array indicating the direction of momentum carried by each
    ray volume: 1 for westerly, 2 for southerly, 3 for easterly, and 4 for
    northerly. Inactive slots are indicated with a zero.

    Parameters
    ----------
    k, l
        Arrays of zonal and meridional wavenumbers.

    Returns
    -------
    np.ndarray
        Integer array indicating the wavenumber quadrant.
    
    """

    return (k > 0) + 2 * (l > 0) + 3 * (k < 0) + 4 * (l < 0)

def _make_callback(
    B: np.ndarray,
    S: np.ndarray,
    wind: np.ndarray
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

        n_seconds = n_step * config.dt
        i = n_seconds // config.dt_output
        i_s = i - int(n_seconds % config.dt_output == 0)

        mom = abs((prop.k + prop.l) * prop.action)
        pdx = _get_quadrant(prop.k, prop.l) - 1
        pdx[prop.m > 0] = -1

        if i_s > -1:
            new = prop.age == 0
            source = (mom * prop.dr)[new]
            np.add.at(S[i_s], pdx[new], source)

        if n_seconds % config.dt_output:
            return

        z = prop._z_padded
        cg = prop._get_cg_r(mean)
        stacked = np.vstack((mom, mom * cg))
        _project(prop.r, prop.dr, z, stacked, pdx, B[i])        
        wind[i] = mean.wind

    return callback

@nb.njit
def _project(
    r: np.ndarray,
    dr: np.ndarray,
    edges: np.ndarray,
    data: np.ndarray,
    pdx: np.ndarray,
    B: np.ndarray,
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
            B[:, p, j] += frac * data[:, i]
