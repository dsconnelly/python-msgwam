import json

from typing import Any, Optional

from msgwam import config

from ..learning.training.io import get_best_trial

def get_overrides(strategy: str, *args: str) -> dict[str, Any]:
    """
    Load the configuration overrides particular to a given strategy. Implemented
    as a standalone function so that other code requiring access to particular
    subfunctions need not import this entire module.

    Parameters
    ----------
    strategy
        Name of the strategy for which to load configuration overrides. There
        must be a function named `_get_{strategy}_overrides` in this namespace.
    args
        Arguments to pass to the override function, if any.

    Returns
    -------
    dict[str, Any]
        Dictionary of settings to be passed to `config.override`.
 
    """

    func_name = f'_get_{strategy}_overrides'
    return globals()[func_name](*args)

def _get_coarse_overrides(
    prune_by: str='flux',
    equal_in: str='flux'
) -> dict[str, Any]:
    """
    Once the configuration file has been updated following the grid search over
    coarse resolutions, the settings for the coarse integration are already set,
    except that we allow some alternate parameters to be varied.

    Parameters
    ----------
    prune_by
        Pruning strategy to use.
    equal_in
        How to discretize the source. Must be either `'cp'`, in which case each
        source ray volume will have equal extent in phase speed spacee, or
        `'flux'`, in which case they will have equal flux.

    """

    equal_flux = {'cp' : False, 'flux' : True}[equal_in]
    return {'prune_by' : prune_by, 'equal_flux' : equal_flux}

def _get_eulerian_overrides() -> dict[str, Any]:
    """Use an Eulerian scheme instead of the ray tracer."""

    return {
        'propagator_type' : 'eulerian',
        'dr_source' : -config.dt,
        'n_source' : 256,
        'dr_min' : 0,
        'n_c' : 30,
        'n_k' : 5,
    }

def _get_ICONlike_overrides() -> dict[str, Any]:
    """Use a configuration similar to that in Bölöni et al. (2020)."""

    return {
        'dr_source' : 1000,
        'n_source' : 24,
        'n_max' : 1250,
    }

def _get_instantaneous_overrides() -> dict[str, Any]:
    """
    Use an instantaneous propagator instead of the ray tracer. The computational
    gains allow us to increase the spectral resolution of the source.
    """

    n_source = _get_reference_overrides()['n_source']
    return {'propagator_type' : 'instantaneous', 'n_source' : n_source}

def _get_MiMAlike_overrides() -> dict[str, Any]:
    """Use a configuration similar to that of online tests in MiMA."""

    return {
        'dr_source' : 1500,
        'n_max' : 2500,
        'n_source' : 40
    }

def _get_network_overrides(
    exp_name: str,
    n_bins: Optional[int]=None
) -> dict[str, Any]:
    """Use a neural network to advance the wave momentum state."""

    if n_bins is None:
        trial = get_best_trial(exp_name)
        i = trial.suggest_int('n_bin_idx', 1, 4)
        n_bins = [1, 2, 3, 4, 5][i]

    else:
        n_bins = int(n_bins)

    return {
        'model_path' : f'data/ml-accel/models/scripted-{exp_name}.jit',
        'propagator_type' : 'network',
        'n_bins' : n_bins,
        'dr_source' : -1200,
        'n_source' : 128,
        'dr_min' : 0
    }

def _get_reference_overrides() -> dict[str, Any]:
    """Integrate at high resolution with no pruning."""

    return {
        'n_max' : 25000,
        'dr_source' : 100,
        'n_source' : 200,
        'n_increment' : 1000,
        'prune_by' : 'none'
    }

def _get_stochastic_overrides(speedup_str: str) -> dict[str, Any]:
    """
    Use a stochastic source instead of a constant-flux source.

    Parameters
    ----------
    speedup_str
        The speedup that should be used. Its square root is the factor by which
        each dimension will be refined at the source. Accepted as a string so
        that this parameter can be provided at the command line.

    """

    speedup = int(speedup_str)
    root = int(speedup ** 0.5)

    return {
        'epsilon' : 1 / speedup,
        'dr_source' : config.dr_source / root,
        'n_source' : int(config.n_source * root),
        'source_type' : 'stochastic'
    }
