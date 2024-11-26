from typing import Any

import numpy as np
import xarray as xr

from msgwam import config
from msgwam.integration import integrate as _integrate
from msgwam.means import PrescribedWind
from msgwam.propagators import TransientPropagator

_N_SAMPLES = 25

def get_overrides(strategy: str) -> dict[str, Any]:
    """
    Load the configuration overrides particular to a given strategy. Implemented
    as a standalone function so that other code need not import this entire
    module for access to the particular subfunctions.

    Parameters
    ----------
    strategy
        Name of the strategy to load configuration overrides for. This module
        must contain a function named `_get_{strategy}_overrides`.

    Returns
    -------
    dict[str, Any]
        Dictionary of settings to be passed to `config.override`.

    """

    func_name = f'_get_{strategy}_overrides'
    return globals()[func_name]()

def integrate(strategy: str) -> None:
    """
    Integrate with configuration settings specific to the given strategy.

    Parameters
    ----------
    strategy
        Name of configuration with which to integrate.

    """

    path = f'data/{config.name}/{strategy}.nc'
    with config.override(**get_overrides(strategy)):
        _get_integration(strategy).to_netcdf(path)

def _get_coarse_overrides() -> dict[str, Any]:
    """
    Once the configuration file has been updated as in `coarsening.py`, the
    settings for the coarse integration are already set.
    """

    return {}

def _get_instantaneous_overrides() -> dict[str, Any]:
    """Use an instantaneous propagator instead of the ray tracer."""

    return {'propagator_type' : 'instantaneous'}

def _get_integration(strategy: str) -> xr.Dataset:
    """
    Integration the solver and return a dataset containing the outputs. Broken
    out as a separate function for easier handling of the stochastic logic.

    Parameters
    ----------
    strategy
        Name of configuration with which to integrate.

    Returns
    -------
    xr.Dataset
        Dataset containing the integration outputs. If integrating with the
        stochastic strategy, the dataset will have a `'sample'` dimension.

    """

    if strategy != 'stochastic':
        return _integrate()
    
    datasets = []
    for i in range(_N_SAMPLES):
        ds = _integrate().assign_coords(sample=i)
        datasets.append(ds)

    return xr.concat(datasets, dim='sample')

def _get_reference_overrides() -> dict[str, Any]:
    """Configuration settings for the high-resolution reference integration."""

    overrides = {
        'dt' : 30,
        'n_grid' : 201,
        'n_source' : 200,
        'n_max' : int(300e3),
        'n_increment' : int(10e3),
        'prune_by' : 'none'
    }

    with config.override(**overrides):
        mean = PrescribedWind()
        cg = TransientPropagator(mean)._get_cg_r(mean)
        dr = np.ceil(np.nanmax(cg * config.dt) / 25) * 25

    overrides['dr_init'] = dr
    return overrides

def _get_stochastic_overrides() -> dict[str, Any]:
    """Use a stochastic source that launches nine times less often."""

    return {
        'epsilon' : 1 / 9,
        'dr_init' : config.dr_init / 3,
        'n_source' : int(config.n_source * 3),
        'source_type' : 'stochastic'
    }