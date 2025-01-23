from typing import Any
from warnings import warn

import numpy as np
import xarray as xr

from msgwam import config
from msgwam.integration import integrate as _integrate
from msgwam.means import InteractiveWind
from msgwam.propagators import TransientPropagator

from .. import hyperparameters as hp

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

    strategy, *args = strategy.split('-')
    func_name = f'_get_{strategy}_overrides'
    return globals()[func_name](*args)

def integrate(strategy: str) -> None:
    """
    Integrate with configuration settings specific to the given strategy.

    Parameters
    ----------
    strategy
        Name of configuration with which to integrate.

    """

    path = f'data/{config.name}/strategies/{strategy}.nc'
    with config.override(**get_overrides(strategy)):
        _get_integration(strategy).to_netcdf(path)

def _get_coarse_overrides() -> dict[str, Any]:
    """
    Once the configuration file has been updated as in `coarsening.py`, the
    settings for the coarse integration are already set.
    """

    return {'jitter' : True}

def _get_instantaneous_overrides() -> dict[str, Any]:
    """Use an instantaneous propagator instead of the ray tracer."""

    return {'propagator_type' : 'instantaneous', 'n_source' : 100}

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

    if strategy == 'reference' or config.propagator_type != 'transient':
        return _integrate()

    datasets = []
    for i in range(hp.strategies.n_samples):
        ds = _integrate().assign_coords(sample=i)
        datasets.append(ds)

    return xr.concat(datasets, dim='sample')

def _get_reference_overrides() -> dict[str, Any]:
    """Configuration settings for the high-resolution reference integration."""

    overrides = {
        'dt' : 30,
        'dr_init' : 50,
        'n_source' : 124,
        'n_max' : int(250e3),
        'n_increment' : 1000,
        'prune_by' : 'none'
    }

    with config.override(**overrides):
        mean = InteractiveWind()
        cg = TransientPropagator(mean)._get_cg_r(mean)
        dr_min = np.ceil(np.nanmax(cg * config.dt) / 25) * 25

        if overrides['dr_init'] < dr_min:
            warn(f'Changing dr to {dr_min}')
            overrides['dr_init'] = dr_min

    return overrides

def _get_surrogate_overrides() -> dict[str, Any]:
    """Use a pretrained surrogate as the propagator."""

    return {
        'n_grid' : 101,
        'propagator_type' : 'network',
        'network_path' : f'data/{config.name}/surrogate-fine/model-best.jit',
        'lookback' : hp.generation.lookback,
        'time_horizon' : config.dt,
        'dr_init' : -1,
    }

def _get_stochastic_overrides(speedup_str: str) -> dict[str, Any]:
    """
    Use a stochastic source that launches less often.

    Parameters
    ----------
    speedup_str
        The speedup that should be targeted. Its square root is the factor by
        which each dimension will be refined. Accepted as a string so that this
        parameter can be provided at the command line.
    
    """

    speedup = int(speedup_str)
    root = int(speedup ** 0.5)

    return {
        'epsilon' : 1 / speedup,
        'dr_init' : config.dr_init / root,
        'n_source' : int(config.n_source * root),
        'source_type' : 'stochastic'
    }
