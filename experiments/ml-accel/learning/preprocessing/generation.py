from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from msgwam import config
from msgwam.integration import integrate
from msgwam.means import MeanState
from msgwam.sources import Source
from msgwam.sources.spectra import _gaussians
from msgwam.utils import shapiro_filter

from ...evaluation.scenarios import _get_descending_jets

from .. import hyperparameters as hp
from ..utils import get_overrides

if TYPE_CHECKING:
    from msgwam.integration import _Callback
    from msgwam.propagators import TransientPropagator

class EnoughPackets(Exception):
    pass

class NotEnoughPackets(Exception):
    pass

def save_training_context() -> None:
    """
    In preparation for training machine learning models, save longer versions of
    the mean wind and source spectrum files, generated with the same processes
    but using different random seeds.
    """

    kwargs = get_overrides()
    kwargs['n_source'] = int(1e3)
    kwargs['spectrum_type'] = 'gaussians'
    kwargs['seed'] = hash(config.name) % 2 ** 32

    with config.override(**kwargs):
        _get_descending_jets(seed=5).to_netcdf(config.prescribed_wind_file)

    kwargs['dt'] = kwargs['dt_launch']
    with config.override(**kwargs):
        _gaussians().to_netcdf(config.spectrum_file)

def save_training_data() -> None:
    """
    Save initial zonal wind profiles, source ray volume properties, and time-
    averaged momentum flux profiles for each fine and coarse packet.
    """

    with config.override(**get_overrides()):
        u, rays = _generate_inputs()
        Y_coarse = _generate_outputs()

    # with config.override(**get_overrides(fine=True)):
        # Y_fine = _generate_outputs()

    np.save(f'data/{config.name}/training/u.npy', u)
    np.save(f'data/{config.name}/training/rays.npy', rays)
    np.save(f'data/{config.name}/training/Y-coarse.npy', Y_coarse)
    # np.save(f'data/{config.name}/training/Y-fine.npy', Y_fine)

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
    rays = np.zeros((hp.n_packets, 7))

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
        rays[i:(i + n_add)] = data.T[:n_add]

        i = i + n_add
        if i == hp.n_packets:
            return u, rays
        
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

    Y = np.zeros((hp.n_packets, config.n_grid))
    callback = _make_callback(Y)

    try:
        _ = integrate(callback)

    except EnoughPackets:
        return config.dt * Y / (hp.max_days * 86400)
    
    raise NotEnoughPackets

def _make_callback(Y: np.ndarray) -> _Callback:
    """
    Make a callback function that stores packet launch times in `starts` and
    flux profiles associated with each packet in `Y`.

    Parameters
    ----------
    Y
        Two-dimensional array whose first dimension ranges over packets and
        whose second dimension ranges over vertical grid cells. Will eventually
        hold time-averaged momentum flux profiles.

    Returns
    -------
    _Callback
        Callback function to pass to `integrate`.

    """

    def callback(
        mean: MeanState,
        prop: TransientPropagator,
        _: int
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
        profiles, = prop._project(flux[None], prop._z_padded, pdx)
        profiles[:, 1:-1] = shapiro_filter(profiles.T).T

        keep = labels < hp.n_packets        
        if keep.sum() == 0:
            raise EnoughPackets
        
        prop._delete_rays(~np.isin(labels[pdx], labels[keep]))
        profiles, labels = profiles[keep], labels[keep]
        Y[labels] += profiles

    return callback