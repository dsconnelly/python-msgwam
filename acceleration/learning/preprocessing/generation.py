from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from msgwam import config
from msgwam.integration import integrate

from ...hyperparameters import generation as hp
from ...shared.distributed import add_task_info
from ...shared.filtering import gaussian_filter
from .overrides import get_overrides

if TYPE_CHECKING:
    from msgwam.integration import _Callback
    from msgwam.means import PrescribedWind
    from msgwam.propagators import TransientPropagator

_SOURCE_PROPS = ['k', 'action']
_STATE_PROPS = ['r', 'k', 'm', 'action', 'age']

def save_training_data():
    """
    Save the mean wind, source, ray volume state, and components of the flux
    profiles. The latter two arrays have one extra time step to represent the
    initializtion of the propagator.
    """

    with config.override(**get_overrides()):
        u = np.zeros((config.n_steps, config.n_grid - 1))
        S = np.zeros((config.n_steps, len(_SOURCE_PROPS), config.n_source))
        R = np.zeros((config.n_steps, len(_STATE_PROPS), config.n_max))
        
        ds = integrate(_make_callback(u, S, R))
        kwargs = {'hours' : hp.filter_hours, 'z_faces' : hp.filter_meters}

        F = np.stack((
            gaussian_filter(ds['pmf_e'], **kwargs),
            gaussian_filter(ds['pmf_w'], **kwargs)
        ), axis=1)

    u, S = u[1:], S[1:]
    for data, name in zip([u, S, R, F], ['u', 'S', 'R', 'F']):
        path = f'data/{config.name}/training/{name}.npy'
        np.save(add_task_info(path), data)

def _make_callback(u: np.ndarray, S: np.ndarray, R: np.ndarray) -> _Callback:
    """
    Make a callback that stores ray volume, source, and zonal wind information
    in the provided arrays.

    Parameters
    ----------
    u
        Two-dimensional array whose first dimension ranges over time steps and
        whose second dimension ranges over vertical grid cell centers.
    S
        Three-dimensional array whose first dimension ranges over time steps,
        whose seccond dimension ranges over source properties, and whose third
        dimension ranges over ray volumes added at the source.
    R
        Three-dimensional array whose first dimension ranges over time steps,
        whose second dimension ranges over ray volume properties, and whose
        third dimension ranges over individual ray volumes.

    Returns
    -------
    _Callback
        Callback function to pass to `integrate`.

    """

    def callback(
        mean: PrescribedWind,
        prop: TransientPropagator,
        n_step: int
    ) -> None:
        """Callback function storing ray volume, source and wind data."""

        _get = lambda name: getattr(prop, name)
        source = np.vstack(list(map(_get, _SOURCE_PROPS)))
        state = np.vstack(list(map(_get, _STATE_PROPS)))
        
        idx = (prop.age == 0) & np.isin(np.arange(config.n_max), prop._ghosts)
        u[n_step] = mean._wind[max(n_step - 1, 0), 0]
        S[n_step, :, :idx.sum()] = source[:, idx]
        R[n_step] = state

    return callback
