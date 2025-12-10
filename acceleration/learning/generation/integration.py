from __future__ import annotations
from typing import TYPE_CHECKING, Optional

import numpy as np
import xarray as xr

from msgwam import config
from msgwam.integration import integrate
from msgwam.utils import get_vertical_grids

from ... import hyperparameters as _hp
from ...hyperparameters import generation as hp

from ..propagators import EulerianPropagator

from .utils import (
    get_info,
    get_overrides
)

if TYPE_CHECKING:
    from msgwam.integration import _Callback
    from msgwam.means import MeanState

_N_K = 3
_N_C = 5

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

        C = np.zeros((n_samples, 3, config.n_grid - 1))
        M, F = np.zeros((2, n_samples, 4, _N_K * _N_C, config.n_grid - 1))
        _ = integrate(_make_callback(C, M, F))

        idx = M > 0
        cg = np.zeros_like(M)
        cg[idx] = F[idx] / M[idx]

    shape = (n_samples, 4, _N_K, _N_C, config.n_grid - 1)
    M, cg = M.reshape(*shape), cg.reshape(*shape)

    data = {
        'time' : seconds.astype(int),
        'quadrant' : np.array(qnames),
        'bin_wvn' : np.arange(_N_K),
        'bin_cpt' : np.arange(_N_C),
        'z_centers' : z_centers,
        'z_faces' : z_faces
    }

    dims = ('time', 'quadrant', 'bin_wvn', 'bin_cpt', 'z_centers')

    data['M_bulk'] = (dims, M)
    data['cg_bulk'] = (dims, cg)

    for i, name in enumerate(['u', 'v', 'N']):
        data[name] = (('time', 'z_centers'), C[:, i])

    year, month, site, lat = get_info(n)
    ds = xr.Dataset(data).assign_attrs(latitude=lat)
    dir_name = f'data/ml-accel/integrations-cg/{year}'
    ds.to_netcdf(f'{dir_name}/{site}-{month}.nc')

def _make_callback(
    C: np.ndarray,
    M: np.ndarray,
    F: np.ndarray
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
        prop: EulerianPropagator,
        n_step: int
    ) -> None:
        """Callback function to return as output."""

        n_seconds = n_step * config.dt
        i = n_seconds // hp.dt_output

        if n_seconds % config.dt_output:
            return
        
        _M, _F = [a.reshape(4, -1, config.n_grid - 1) for a in prop._cache]
        edges_k, edges_c = EulerianPropagator._init_edges(_N_K, _N_C)

        wdx = np.digitize(prop._wvn.flatten(), edges_k) - 1
        jdx = np.digitize(prop._cpt.flatten(), edges_c) - 1
        fdx = (wdx[:, None] * _N_C + jdx[None]).ravel()

        for q in range(4):
            np.add.at(M[i, q], fdx, _M[q])
            np.add.at(F[i, q], fdx, _F[q])

        mean.step(None, max(n_step - 1, 0))

        C[i, :2] = mean.wind
        C[i, 2] = mean.N

        mean.step(None, n_step)

    return callback
