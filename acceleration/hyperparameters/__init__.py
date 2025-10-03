import os
import tomllib

from typing import Any, Optional
from warnings import warn

from numpy import meshgrid, stack

from . import architectures
from . import generation
from . import scenarios
from . import strategies
from . import training

__all__ = [
    'architectures',
    'generation',
    'grid_path',
    'scenarios',
    'strategies',
    'training',
    'task_id'
]

_OPTIONS: list[str] = []

grid_path: str
grid_size: int
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
        i = int(os.environ.get('SLURM_ARRAY_TASK_ID', 8))

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
    globals()['_OPTIONS'] = list(options.keys())
    globals()['grid_size'] = params.shape[1]

    if i >= params.shape[1]:
        warn('more jobs than hyperparameter settings')
        i = params.shape[1] - 1

    for name, value in zip(options.keys(), params[:, i]):
        _set_hyperparameter(*name.split('.'), value)    

def show_hyperparameters() -> None:
    """Print the currently loaded set of hyperparameters."""

    to_print = {}
    for name in _OPTIONS:
        sub_name, var_name = name.split('.')
        value = getattr(globals()[sub_name], var_name)
        to_print.setdefault(sub_name, {})[var_name] = value

    for sub_name, params in to_print.items():
        print(f'==== {sub_name} ====')

        for var_name, value in params.items():
            print(f'{var_name}: {value}')

        print()

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
            if var_name.endswith('_t'):
                var_name = var_name[:-2]
                value = tuple(value)

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
