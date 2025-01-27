from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from msgwam import config
from msgwam.integration import integrate
from msgwam.means import PrescribedWind
from msgwam.sources import Source
from msgwam.sources.spectra import _gaussians
from msgwam.utils import shapiro_filter

from ...evaluation.scenarios import _get_descending_jets

from ... import hyperparameters as hp
from ..utils import (
    add_task_info,
    get_generation_mode,
    get_overrides,
    get_workload,
    make_seed
)

if TYPE_CHECKING:
    from msgwam.integration import _Callback
    from msgwam.means import MeanState
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
    
    wind_seed = make_seed(config.name, 'wind', hp.task_id)

    with config.override(**get_overrides()):
        ds = _get_descending_jets(seed=wind_seed)
        ds.to_netcdf(config.prescribed_wind_file)

    if get_generation_mode() == 'tr':
        _set_training_hyperparameters()

        with config.override(**get_overrides(fine=True)):
            ds = _get_descending_jets(seed=wind_seed)
            ds.to_netcdf(config.prescribed_wind_file)

    kwargs = get_overrides()
    kwargs['n_source'] = int(1e3)
    kwargs['spectrum_type'] = 'gaussians'
    kwargs['seed'] = make_seed(config.name, 'spectrum', hp.task_id)
    kwargs['dt'] = kwargs['dt_launch']

    with config.override(**kwargs):
        _gaussians().to_netcdf(config.spectrum_file)

def save_training_data() -> None:
    """
    Save initial zonal wind profiles, source ray volume properties, and time-
    averaged momentum flux profiles for each fine and coarse packet.
    """

    start, end = get_workload(hp.generation.n_packets)
    n_packets = end - start

    with config.override(**get_overrides()):
        u, rays = _generate_inputs(n_packets)
        Y_coarse = _generate_outputs(n_packets)

    with config.override(**get_overrides(fine=True)):
        Y_fine = _generate_outputs(n_packets)

    datas = [u, rays, Y_coarse, Y_fine]
    names = ['u', 'rays', 'flux-coarse', 'flux-fine']
    keep = np.isnan(u).sum(axis=(1, 2)) == 0

    mode = get_generation_mode()
    for data, name in zip(datas, names):
        path = f'data/{config.name}/training/{name}-{mode}.npy'
        np.save(add_task_info(path), data[keep])

def _generate_inputs(n_packets: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate input wind profiles and (coarse) ray volume properties.

    Parameters
    ----------
    n_packets
        How many packets to generate.

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

    shape = (n_packets, 3, config.n_grid - 1)
    u = np.full(shape, fill_value=np.nan)
    rays = np.zeros((n_packets, 7))

    mean = PrescribedWind()
    source = Source.from_name('packet')

    i = 0
    for n_step in range(config.n_steps):
        if n_step * config.dt % config.dt_launch != 0:
            continue

        mean.step(None, n_step)
        data, _ = source.launch(mean, n_step)
        n_add = min(data.shape[1], n_packets - i)
        rays[i:(i + n_add)] = data.T[:n_add]

        n_back = hp.generation.lookback // config.dt
        n_ahead = hp.generation.lookahead // config.dt
        steps = [n_step - n_back, n_step, n_step + n_ahead]

        for k, step in enumerate(steps):
            if 0 <= step < mean._wind.shape[0]:
                u[i:(i + n_add), k] = mean._wind[step, 0]

        i = i + n_add
        if i == n_packets:
            return u, rays
        
    raise NotEnoughPackets

def _generate_outputs(n_packets: int) -> np.ndarray:
    """
    Generate momentum flux profiles averaged over the lifetime of each packet.

    Parameters
    ----------
    n_packets
        How many packets to generate.

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

    Y = np.zeros((n_packets, config.n_grid))
    callback = _make_callback(Y)

    try:
        _ = integrate(callback)

    except EnoughPackets:
        return config.dt * Y / (hp.generation.max_days * 86400)
    
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

        keep = labels < Y.shape[0]
        if keep.sum() == 0:
            raise EnoughPackets
        
        prop._delete_rays(~np.isin(labels[pdx], labels[keep]))
        profiles, labels = profiles[keep], labels[keep]
        Y[labels] += profiles

    return callback

def _set_training_hyperparameters() -> None:
    """
    Temporarily overwrite the hyperparameters governing the training mean state,
    so that different configurations can be used during training and testing.
    """

    # TODO: get these from hyperparameters instead
    *_, period, noise = config.name.split('-')
    a, b = {'fast' : (3, 3), 'slow' : (14, 14), 'variable' : (1, 5)}[period]
    noise = float(noise)

    hp.evaluation.osc_period_min = a
    hp.evaluation.osc_period_max = b
    hp.evaluation.u_noise_scale = noise
    print('set generation hyperparameters:', a, b, noise)