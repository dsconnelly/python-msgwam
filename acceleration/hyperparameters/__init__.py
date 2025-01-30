import os
import tomllib

from typing import Any, Optional
from warnings import warn

from numpy import meshgrid, stack

from . import scenarios
from . import strategies

__all__ = [
    'grid_path',
    'scenarios',
    'strategies',
    'task_id'
]

grid_path: str
task_id: int

def load(path: str, i: Optional[int]=None) -> None:
    """
    Load hyperparameter settings from a TOML file. Parameters that are passed as
    lists of values will be assigned based on the current Slurm task ID. Assigns
    each values to the appropriate submodule with manipulation of `globals()`
    that is not for the faint of heart.

    Parameters
    ----------
    path
        Path to hyperparameter grid file.
    i
        Index into flattened array, denoting the combination of hyperparameters
        to be used. If `None`, determined by the Slurm task.

    """

    if i is None:
        i = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))

    with open(path, 'rb') as f:
        options, constants = _parse_grid(tomllib.load(f))

    globals()['grid_path'] = path
    globals()['task_id'] = i

    for name, value in constants.items():
        _set_hyperparameter(*name.split('.'), value)

    if not options:
        return
    
    mesh = meshgrid(*options.values(), indexing='ij')
    params = stack(mesh, axis=0).reshape(len(options), -1)

    if i >= params.shape[1]:
        warn('more jobs than hyperparameter settings')
        i = params.shape[1] - 1

    for name, value in zip(options.keys(), params[:, i]):
        _set_hyperparameter(*name.split('.'), value)    

def _parse_grid(grid: dict[str, Any]) -> tuple[dict[str, list], dict[str, Any]]:
    """
    Parse a hyperparameter dictionary loaded from a TOML file, separating those
    parameters that are specified as constants and those that should be used to
    form the grid during the hyperparameter sweep.

    Parameters
    ----------
    grid
        Hyperparameter dictionary, as returned by `tomllib.load`.

    Returns
    -------
    dict[str, list]
        Dictionary of hyperparameters with multiple options.
    dict[str, Any]
        Dictionary of hyperparameters set as constants.

    """

    options, constants = {}, {}
    for sub_name, subgrid in grid.items():
        for var_name, value in subgrid.items():
            key = f'{sub_name}.{var_name}'

            if isinstance(value, list):
                options[key] = value

            else:
                constants[key] = value

    return options, constants

def _set_hyperparameter(sub_name: str, var_name: str, value: Any) -> None:
    """
    Set a hyperparameter in a given submodule.

    Parameters
    ----------
    sub_name
        Name of the submodule within which to assign.
    var_name
        Name of the hyperparameter to assign.
    value
        Value to assign to the selected hyperparameter.

    """

    submodule = globals()[sub_name]
    value = submodule.__annotations__[var_name](value)
    setattr(submodule, var_name, value)
