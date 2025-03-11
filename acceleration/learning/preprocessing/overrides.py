from math import ceil
from typing import Any

from msgwam import config

from ...hyperparameters import generation as hp
from ...shared.distributed import N_TASKS, add_task_info

def get_overrides() -> dict[str, Any]:
    """
    Get the configuration overrides for generating machine learning data.

    Returns
    -------
    dict[str, Any]
        Dictionary to pass to `config.override`.

    """

    mean_path = f'data/{config.name}/input/mean-state-training.nc'
    spectrum_path = f'data/{config.name}/input/spectrum-training.nc'
    n_day = _get_n_day()

    return {
        'prescribed_wind_file' : add_task_info(mean_path),
        'spectrum_file' : add_task_info(spectrum_path),
        'n_day' : n_day
    }

def _get_n_day() -> int:
    """
    Get the number of days the integration should last to generate the required
    number of samples on this task.

    Returns
    -------
    int
        Number of days to integrate for.

    """

    return ceil(config.dt * hp.n_samples / N_TASKS / 86400)
