from time import time as now
from typing import Any

import numpy as np
import xarray as xr

from . import config
from .means import MeanState
from .propagators import Propagator, TransientPropagator
from .utils import get_iterator, get_time

def integrate() -> xr.Dataset:
    """
    Integrate the system using the loaded configuration settings.

    Returns
    -------
    xr.Dataset
        Dataset holding integrated mean wind and momentum flux profiles.

    """

    mean = MeanState.from_name(config.mean_state_type)
    prop = Propagator.from_name(config.propagator_type, mean)
    ds = _update_dataset(mean, prop, _init_dataset(mean, prop), 0)

    start = now()
    for n_step in get_iterator():
        mean, prop = mean.step(prop, n_step), prop.step(mean, n_step)
        ds = _update_dataset(mean, prop, ds, n_step)

    runtime = now() - start
    ds = ds.assign_attrs(runtime=runtime)

    return ds

def _init_dataset(mean: MeanState, prop: Propagator) -> xr.Dataset:
    """
    Initialize the dataset to hold the outputted data.

    Parameters
    ----------
    mean
        Initial mean state of the system.
    prop
        Gravity wave propagator to be used.

    Returns
    -------
    xr.Dataset
        Initialized dataset.

    """
    
    data: dict[str, Any] = {
        'time' : get_time()[::config.n_skip],
        'z_centers' : mean.z_centers,
        'z_faces' : mean.z_faces
    }

    for name in ['u', 'v']:
        shape = (len(data['time']), len(data['z_centers']))
        data[name] = (('time', 'z_centers'), np.zeros(shape))

    for name in ['pmf_e', 'pmf_w', 'pmf_n', 'pmf_s']:
        shape = (len(data['time']), len(data['z_faces']))
        data[name] = (('time', 'z_faces'), np.zeros(shape))

    if isinstance(prop, TransientPropagator):
        data['n_rays'] = ('time', np.zeros(len(data['time'])))

    return xr.Dataset(data)

def _update_dataset(
    mean: MeanState,
    prop: Propagator,
    ds: xr.Dataset,
    n_step: int
) -> xr.Dataset:
    """
    Update the dataset with the current system state.

    Parameters
    ----------
    mean
        Current mean state of the system.
    prop
        Gravity wave propagator.
    ds
        Partially-filled dataset.
    n_step
        Index of the current time step.

    Returns
    -------
    xr.Dataset
        Updated dataset.

    """

    k = (n_step - 1) // config.n_skip + 1
    rollover = n_step % config.n_skip == 0

    if not (rollover or config.average_output):
        return ds

    names = ['u', 'v', 'pmf_e', 'pmf_w', 'pmf_n', 'pmf_s']
    profiles = [*mean.wind, *prop.get_fluxes(mean, net=False)]
    factor = 1 / config.n_skip if config.average_output else 1

    for name, profile in zip(names, profiles):
        ds[name][k] = ds[name].values[k] + factor * profile

    if rollover and isinstance(prop, TransientPropagator):
        ds['n_rays'][k] = prop.n_active

    return ds
