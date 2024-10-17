import tomllib

import torch

# global
grid_path: str
task_id: int

# training
batch_size: int
learning_rate: float
noise_level: float

# architectures
conservative: int
batch_norm_pos: int
dropout_rate: float
n_blocks: int
network_size: int

def load(path: str, i: int=0, verbose: bool=True) -> None:
    """
    Load the hyperparameter settings for a particular grid file and Slurm task
    id. Involves manipulation of `globals()` that is not for the faint of heart.

    Parameters
    ----------
    path
        Path to hyperparameter grid file.
    i
        Index into flattened array. Used by job arrays.
    verbose
        Whether to print the hyperparameters after they are set.

    """

    globals()['grid_path'] = path
    globals()['task_id'] = i

    with open(path, 'rb') as f:
        grid = tomllib.load(f)

    func = lambda a: torch.as_tensor(a, dtype=torch.double)
    mesh = torch.meshgrid(*map(func, grid.values()), indexing='ij')
    params = torch.stack(mesh, dim=0).reshape(len(grid), -1)

    show = print if verbose else lambda *_: None
    show(f'==== hyperparameters (task {i}) ====')

    try:
        for name, value in zip(grid.keys(), params[:, i]):
            caster = globals()['__annotations__'][name]
            globals()[name] = caster(value.item())
            show(f'{name} = {globals()[name]}')
    
    except IndexError:
        print('Warning: could not set non-global hyperparameters')

    show()
