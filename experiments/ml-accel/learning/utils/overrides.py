from functools import wraps
from typing import Any, Callable

import numpy as np

from msgwam import config

from ...hyperparameters import generation as hp

from .distributed import N_TASKS, add_task_info, get_generation_mode

def get_overrides(fine: bool=False) -> dict[str, Any]:
    """
    Get the configuration overrides for various machine learning steps.

    Parameters
    ----------
    fine
        Whether to include overrides for the reference fine integration or, if
        `False`, for the coarse integration.

    Returns
    -------
    dict[str, Any]
        Dictionary to pass to `config.override`.

    """

    mean_path = f'data/{config.name}/input/descending-jets.nc'
    spectrum_path = f'data/{config.name}/input/spectrum-training.nc'
    n_day = _get_n_day()

    kwargs = {
        'source_type' : 'packet',
        'prescribed_wind_file' : add_task_info(mean_path),
        'spectrum_file' : add_task_info(spectrum_path),

        'dt' : 30,
        'dt_launch' : hp.dt_launch,
        'dt_output' : n_day * 86400,
        'n_day' : n_day,
        'n_grid' : 101,

        'max_age' : hp.max_days * 86400,
        'n_increment' : 1000,
        'prune_by' : 'none'
    }

    if not fine:
        kwargs['n_chromatic'] = 1
        kwargs['n_repeat'] = 1

        return kwargs
    
    root = int(hp.speedup ** 0.5)
    kwargs['n_source'] = config.n_source * root
    kwargs['dr_init'] = config.dr_init / root
    kwargs['n_chromatic'] = hp.speedup
    kwargs['n_repeat'] = root

    return kwargs

def with_overrides(func: Callable=None, *, fine: bool=False) -> Callable:
    """
    Decorator to specify that a function should be called with the overrides
    returned by `get_overrides`. The confusing nested nature of this function is
    so that it can be called without parentheses when if `not Fine`. Credit to

        https://stackoverflow.com/questions/52126071/
        decorator-with-arguments-avoid-parenthesis-when-no-arguments

    for this design pattern.

    Parameters
    ----------
    func
        Function to be called with overrides, if provided.
    fine
        Argument to pass to `get_overrides` when wrapping the function.

    Returns
    -------
    Callable
        Either the decorator that does of wrapping of a function, if `func` is
        `None`, or the wrapped function of itself, if a function is provided.

    """

    def decorate(func: Callable) -> Callable:
        """Decorator that wraps a function with `config.override`."""

        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            with config.override(get_overrides(fine=fine)):
                return func(*args, **kwargs)
            
        return wrapper
    
    if func is None:
        return decorate
    
    return decorate(func)

def _get_n_day() -> int:
    """
    Get a reasonable upper bound on the number of days the integration should
    run to generate the number of packets requested. Adjusts this number in the
    case where the generation process is parallelized.

    Returns
    -------
    int
        Number of days to be prepared to integrate for.

    """

    last_start = hp.dt_launch * hp.n_packets / config.n_source / N_TASKS
    return int(last_start / 86400 + hp.max_days) + 5
