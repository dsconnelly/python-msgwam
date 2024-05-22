import os

import torch

beta: float
learning_rate: float
network_size: int
smoothing: float
weight_decay: float

_grid = {
    'beta' : [0.1, 0.8],
    'learning_rate' : [1e-3, 2e-4],
    'network_size' : [2, 5],
    'smoothing' : [4, 9],
    'weight_decay' : [0, 1e-4]
}

def _set_hyperparameters():
    """
    Set the hyperparameters according to the Slurm array index. Involves some
    manipulation of `globals()` which are not for the faint of heart. Also
    prints the hyperparameter choices.
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
