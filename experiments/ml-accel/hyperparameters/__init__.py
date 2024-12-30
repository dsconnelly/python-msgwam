import os
import tomllib

from numpy import meshgrid, stack
from typing import Optional
from warnings import warn

from . import architectures, evaluation, generation, training

grid_path: str
task_id: int

__all__ = [
    'architectures',
    'generation',
    'grid_path',
    'load',
    'task_id',
    'training'
]

def load(path: str, i: Optional[int]=None, verbose: bool=True) -> None:
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
    verbose
        Whether to print the hyperparameters after assignment.
    
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

    last_name = None
    for name, value in zip(names, params[:, i]):
        sub_name, var_name = name.split('.')
        submodule = globals()[sub_name]

        value = submodule.__annotations__[var_name](value)
        setattr(submodule, var_name, value)

        if verbose:
            if (last_name is not None) and (last_name != sub_name):
                print()
            
            print(f'{name} = {value}')
            last_name = sub_name
    