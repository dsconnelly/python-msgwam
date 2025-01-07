import os
import tomllib

from numpy import meshgrid, stack
from types import ModuleType
from typing import Optional
from warnings import warn

from . import architectures, evaluation, generation, strategies, training

grid_path: str
task_id: int

__all__ = [
    'architectures',
    'display',
    'evaluation',
    'generation',
    'grid_path',
    'load',
    'task_id',
    'strategies'
    'training'
]

def display() -> None:
    """Display the currently-loaded hyperparameter settings."""

    for sub_name in __all__:
        submodule = globals()[sub_name]
        if not isinstance(submodule, ModuleType):
            continue

        for name in submodule.__annotations__:
            print(f'{sub_name}.{name} = {getattr(submodule, name)}')

        print()

def load(path: str, i: Optional[int]=None) -> None:
    """
    Load hyperparameter settings from a TOML file. Parameters that are passed as
    lists of values instead of individual values will be assigned based on the
    current Slurm task ID. Assigns the values to the appropriate submodules, and
    involves manipulation of `globals()` that is not for the faint of heart.

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

    globals()['grid_path'] = path
    globals()['task_id'] = i

    with open(path, 'rb') as f:
        grid = tomllib.load(f)

    names, to_mesh = [], []
    for sub_name, subgrid in grid.items():
        for var_name, value in subgrid.items():
            if not isinstance(value, list):
                value = [value]

            names.append(f'{sub_name}.{var_name}')
            to_mesh.append(value)

    mesh = meshgrid(*to_mesh, indexing='ij')
    params = stack(mesh, axis=0).reshape(len(to_mesh), -1)

    if i >= params.shape[1]:
        warn('more jobs than hyperparameter settings')
        i = 0

    for name, value in zip(names, params[:, i]):
        sub_name, var_name = name.split('.')
        submodule = globals()[sub_name]

        value = submodule.__annotations__[var_name](value)
        setattr(submodule, var_name, value)
