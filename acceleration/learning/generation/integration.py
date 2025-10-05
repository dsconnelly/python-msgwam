from __future__ import annotations
from typing import TYPE_CHECKING, Optional

import numba as nb
import numpy as np
import xarray as xr

from msgwam import config
from msgwam.integration import integrate
from msgwam.utils import get_vertical_grids

from ... import hyperparameters as _hp
from ...hyperparameters import generation as hp

from .utils import get_bin_edges, get_overrides, get_pdx, get_site_and_lat

if TYPE_CHECKING:
    from msgwam.integration import _Callback
    from msgwam.means import MeanState
    from msgwam.propagators import TransientPropagator

def save_training_data(n_str: Optional[str]=None) -> None:
    """
    Integrate and save the relevant quantities for training.
    
    Parameters
    ----------
    n_str
        Task ID number for which to save training data. If not provided, uses
        the current SLURM job array step. Passed as a string since this is a
        command line argument.

    """

    n = _hp.task_id
    if n_str is not None:
        n = int(n_str)

    with config.override(**get_overrides(n)):
        n_samples = 1 + (86400 * config.n_day) // config.dt_output
        seconds = np.arange(n_samples) * config.dt_output
        qnames = ['k > 0', 'l > 0', 'k < 0', 'l < 0']
        z_faces, z_centers = get_vertical_grids()

        windN = np.zeros((n_samples, 3, config.n_grid - 1))
        M, S = np.zeros((2, n_samples, 4 * hp.n_bins, config.n_grid - 1))
        D = np.zeros((n_samples, 4, config.n_grid - 1))
        F = np.zeros((n_samples, 4, config.n_grid))

        _ = integrate(_make_callback(windN, M, D, S, F))
        args = (n_samples, 4, hp.n_bins, config.n_grid - 1)
        M, S = M.reshape(*args), S.reshape(*args)

    data = {
        'time' : seconds.astype(int),
        'quadrant' : np.array(qnames),
        'bin' : np.arange(hp.n_bins),
        'z_centers' : z_centers,
        'z_faces' : z_faces
    }

    edges = get_bin_edges()
    data['bin_center'] = (('bin'), (edges[:-1] + edges[1:]) / 2)
    data['bin_width'] = (('bin'), edges[1:] - edges[:-1])

    data['M_bulk'] = (('time', 'quadrant', 'bin', 'z_centers'), M)
    data['source'] = (('time', 'quadrant', 'bin', 'z_centers'), S)
    data['sink'] = (('time', 'quadrant', 'z_centers'), D)
    data['F_bulk'] = (('time', 'quadrant', 'z_faces'), F)
    
    for i, name in enumerate(['u', 'v', 'N']):
        data[name] = (('time', 'z_centers'), windN[:, i])

    site, lat = get_site_and_lat(n // 12)
    ds = xr.Dataset(data).assign_attrs(latitude=lat)
    ds.to_netcdf(f'data/ml-accel/training/{site}-{(n % 12) + 1}.nc')

def _make_callback(
    windN: np.ndarray,
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

        n_seconds = n_step * config.dt
        i = n_seconds // hp.dt_output
        i = i + bool(n_seconds % hp.dt_output)

        wvn = abs(prop.k + prop.l)
        mom = wvn * prop.action

        cp_hat = prop._get_omega_hat(mean) / wvn
        bdx = get_pdx(prop.k, prop.l, cp_hat)
        pdx = bdx // hp.n_bins

        since_last = n_seconds - (i - 1) * config.dt_output
        attr = prop.attrition * (abs(prop.age) >= since_last)   
        cg = prop._get_cg_r(mean) / (hp.dt_output // hp.dt)

        _project(prop.r, prop.dr, mean.z_faces, attr, pdx, D[i])
        _project(prop.r, prop.dr, prop._z_padded, mom * cg, pdx, F[i])
        _break_oob_rays(prop, mean, mom + attr, pdx, D[i, :, -config.n_sponge:])
        
        if n_seconds % hp.dt_output:
            return

        windN[i, :2] = mean.wind
        windN[i, 2] = mean.N

        ndx = bdx.copy()
        ndx[prop.age >= hp.dt_output] = -1
        _project(prop.r, prop.dr, mean.z_faces, mom, bdx, M[i])
        _project(prop.r, prop.dr, mean.z_faces, mom, ndx, S[i])

    return callback

def _break_oob_rays(
    prop: TransientPropagator,
    mean: MeanState,
    mom: np.ndarray,
    pdx: np.ndarray,
    D: np.ndarray
) -> None:
    """
    Catch the contributions from ray volumes that have partially or fully exited
    the upper boundary, and add their momentum into the sponge layer.
    """
    
    r_lo = prop.r - 0.5 * prop.dr
    r_hi = prop.r + 0.5 * prop.dr

    dr = np.maximum(r_hi - np.maximum(r_lo, config.z_max), 0)
    dz = mean.z_faces[-1] - mean.z_faces[-(config.n_sponge + 1)]
    sponged = (dr * mom / dz)[prop._valid, None]
    np.add.at(D, pdx[prop._valid], sponged)

    prop._data[1] = prop.dr - dr
    prop._data[0] = r_lo + prop.dr / 2
    prop._delete_rays(prop.age < 0)

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
