from typing import Any

from msgwam import config

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

def _get_coarse_overrides() -> dict[str, Any]:
    """
    Once the configuration file has been updated following the grid search over
    coarse resolutions, the settings for the coarse integration are already set,
    except that the integration should be performed with initial jitter.
    """

    return {'jitter' : True}

def _get_ICONlike_overrides() -> dict[str, Any]:
    """Use a configuration similar to that in Bölöni et al. (2020)."""

    return {'dr_init' : 1000, 'n_source' : 48, 'n_max' : 2500, 'jitter' : True}

def _get_instantaneous_overrides() -> dict[str, Any]:
    """
    Use an instantaneous propagator instead of the ray tracer. The computational
    gains allow us to increase the spectral resolution of the source.
    """

    return {'propagator_type' : 'instantaneous', 'n_source' : 120}

def _get_reference_overrides() -> dict[str, Any]:
    """Integrate at high resolution with no pruning."""

    return {
        'dt' : 30,
        'dr_init' : 50,
        'n_source' : 144,
        'n_max' : int(250e3),
        'n_increment' : 1000,
        'prune_by' : 'none',
        'max_dt_multiplier' : 6
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
        'dr_init' : config.dr_init / root,
        'n_source' : int(config.n_source * root),
        'source_type' : 'stochastic'
    }