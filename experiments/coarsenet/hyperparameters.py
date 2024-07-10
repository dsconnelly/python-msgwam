import os

import torch

conservative: bool
learning_rate: float
network_size: int
root: int
use_adam: bool
weight_decay: float

_grid = {
    'conservative' : [1, 0],
    'learning_rate' : [1e-4, 1e-5],
    'network_size' : [3, 6],
    'root' : [5],
    'use_adam' : [1, 0],
    'weight_decay' : [0, 0.1]
}

def _set_hyperparameters():
    """
    Set the hyperparameters according to the Slurm array index. Involves some
    manipulation of `globals()` which is not for the faint of heart. Also prints
    the hyperparameter choices.
    """

    func = lambda a: torch.as_tensor(a, dtype=torch.double)
    mesh = torch.meshgrid(*map(func, _grid.values()), indexing='ij')
    params = torch.stack(mesh, dim=0).reshape(len(_grid), -1)

    print('==== hyperparameter values ====')
    for name, value in zip(_grid.keys(), params[:, task_id]):
        caster = globals()['__annotations__'][name]
        globals()[name] = caster(value.item())
        print(f'{name} = {globals()[name]}')

    print()

task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
_set_hyperparameters()
