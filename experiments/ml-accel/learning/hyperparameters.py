import os
import tomllib

from typing import Optional

import numpy as np

################################################################################
# global
################################################################################
grid_path: str
task_id: int

################################################################################
# generation
################################################################################
max_days: int
n_packets: int
speedup: int

################################################################################
# architectures
################################################################################
batch_norm_pos: int
dropout_rate: float
layer_size: int
n_blocks: int
n_layers: int

def load(path: str, i: Optional[int]=None, verbose: bool=False) -> None:
    """
    Load the hyperparameter settings for a particular grid file and Slurm task
    ID. Involves manipulation of `globals()` that is not for the faint of heart.

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

    mesh = np.meshgrid(*grid.values(), indexing='ij')
    params = np.stack(mesh, axis=0).reshape(len(grid), -1)

    for name, value in zip(grid.keys(), params[:, i]):
        caster = __annotations__[name]
        globals()[name] = caster(value)

    if verbose:
        _display_hyperparameters()

def _display_hyperparameters() -> None:
    """Display the loaded values of each hyperparameter."""

    for name in __annotations__.keys():
        if name == 'grid_path':
            continue
        
        print(f'{name} = {globals()[name]}')