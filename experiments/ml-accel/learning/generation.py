from __future__ import annotations
from typing import TYPE_CHECKING, Any

import numpy as np

from msgwam import config
from msgwam.integration import integrate
from msgwam.means import MeanState
from msgwam.sources import Source
from msgwam.utils import shapiro_filter

from . import hyperparameters as hp

if TYPE_CHECKING:
    from msgwam.integration import _Callback
    from msgwam.propagators import TransientPropagator

class EnoughPackets(Exception):
    pass

class NotEnoughPackets(Exception):
    pass

def save_training_data() -> None:
    """
    Save initial zonal wind profiles, source ray volume properties, and time-
    averaged momentum flux profiles for each fine and coarse packet.
    """

    with config.override(**_get_overrides()):
        u, X = _generate_inputs()
        Y_coarse = _generate_outputs()

    with config.override(**_get_overrides(fine=True)):
        Y_fine = _generate_outputs()

    np.save(f'data/{config.name}/u.npy', u)
    np.save(f'data/{config.name}/X.npy', X)
    np.save(f'data/{config.name}/Y-fine.npy', Y_fine)
    np.save(f'data/{config.name}/Y-coarse.npy', Y_coarse)

def _get_overrides(fine: bool=False) -> dict[str, Any]:
    """
    Get the configuration overrides to generate data at a specified resolution.
    
    Parameters
    ----------
    fine
        Whether to get overrides for the reference fine integration or, if
        `False`, for the coarse integration.

    Returns
    -------
    dict[str, Any]
        Dictionary to pass to `config.override`.

    """

    root = int(hp.speedup ** 0.5)
    seed = hash(config.name) % 2 ** 32

    kwargs = {
        'source_type' : 'packet',
        'spectrum_type' : 'gaussians',
        'dt_launch' : 6 * 3600,
        'n_increment' : 1000,
        'prune_by' : 'none',
        'seed' : seed
    }

    if not fine:
        kwargs['n_chromatic'] = 1
        kwargs['n_repeat'] = 1

        return kwargs

    kwargs['n_source'] = config.n_source * root
    kwargs['n_max'] = config.n_max * hp.speedup
    kwargs['dr_init'] = config.dr_init / root
    kwargs['n_chromatic'] = hp.speedup
    kwargs['n_repeat'] = root

    return kwargs

def _generate_inputs() -> tuple[np.ndarray, np.ndarray]:
    """
    Generate input wind profiles and (coarse) ray volume properties.

    Returns
    -------
    np.ndarray
        Array of initial zonal wind profiles.
    np.ndarray
        Array of ray volume properties excluding those of vertical position.

    Raises
    ------
    NotEnoughPackets
        If the integration is too short to generate sufficiently many packets.

    """

    u = np.zeros((hp.n_packets, config.n_grid - 1))
    X = np.zeros((hp.n_packets, 7))

    mean = MeanState.from_name('prescribed')
    source = Source.from_name('packet')

    i = 0
    for n_step in range(config.n_steps):
        if n_step * config.dt % config.dt_launch != 0:
            continue

        mean.step(None, n_step)
        data, _ = source.launch(mean, n_step)
        n_add = min(data.shape[1], hp.n_packets - i)

        u[i:(i + n_add)] = mean.u
        X[i:(i + n_add)] = data.T[:n_add]

        i = i + n_add
        if i == hp.n_packets:
            return u, X
        
    raise NotEnoughPackets

def _generate_outputs() -> np.ndarray:
    """
    Generate momentum flux profiles averaged over the lifetime of each packet.

    Returns
    -------
    np.ndarray
        Two-dimensional array whose first dimension ranges over packets, and
        whose second dimension ranges over vertical grid faces.

    Raises
    ------
    NotEnoughPackets
        If the integration is too short to generate sufficiently many packets.

    """

    starts = np.inf * np.ones(hp.n_packets)
    Y = np.zeros((config.n_steps, hp.n_packets, config.n_grid))
    callback = _make_callback(starts, Y)

    try:
        _ = integrate(callback)

    except EnoughPackets:
        return config.dt * Y.sum(axis=0) / (hp.max_days * 86400)
    
    raise NotEnoughPackets

def _make_callback(starts: np.ndarray, Y: np.ndarray) -> _Callback:
    """
    Make a callback function that stores packet launch times in `starts` and
    flux profiles associated with each packet in `Y`.

    Parameters
    ----------
    starts
        Array with an entry for each packet to be launched.
    Y
        Three-dimensional array whose first dimension ranges over time steps,
        whose seocnd dimension ranges over packets, and whose third dimension
        ranges over vertical grid cells. Will hold flux profiles.

    Returns
    -------
    _Callback
        Callback function to pass to `integrate`.

    """

    def callback(
        mean: MeanState,
        prop: TransientPropagator,
        n_step: int
    ) -> None:
        """
        Callback function storing appropriate data in `starts` and `Y`.

        Raises
        ------
        EnoughPackets
            When enough packets have been generated, to signal that that the
            integration need continue no further.
        
        """

        labels, pdx = prop._get_packet_info()
        flux = prop.k * prop.action * prop._get_cg_r(mean)
        profiles = prop._project(flux[None], prop._z_padded, pdx)[0]

        profiles = profiles[labels < hp.n_packets]
        labels = labels[labels < hp.n_packets]

        new = np.isinf(starts[labels])
        starts[labels[new]] = n_step

        keep = config.dt * (n_step - starts[labels]) < hp.max_days * 86400
        labels, profiles = labels[keep], profiles[keep]

        if keep.sum() == 0:
            raise EnoughPackets
        
        profiles[:, 1:-1] = shapiro_filter(profiles.T).T
        Y[n_step][labels] = profiles

    return callback