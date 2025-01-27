from __future__ import annotations
from typing import Literal, Optional

import numpy as np
import torch

from msgwam import config

from ... import hyperparameters as hp
from .distributed import add_task_info

def load_data(
    target_type: str,
    dataset: Literal['tr', 'te'],
    n_samples: Optional[int]=None,
    distributed: bool=False,
    **kwargs
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load machine learning input and output data from disk.

    Parameters
    ----------
    target_type
        Kind of target data to load, as passed to `train_network`.
    dataset
        Whether to load from `'tr'` or `'te'` data.
    n_samples
        How many samples to load. If `None`, load all available data.
    distributed
        If `True`, only load the data generated on the current Slurm task ID.
    **kwargs
        Keyword arguments for the specified target type.

    Returns
    -------
    torch.Tensor
        Loaded zonal wind profiles.
    torch.Tensor
        Loaded ray volume properties.
    torch.Tensor
        Loaded targets.

    """

    def load(base: str) -> torch.Tensor:
        mode = None if n_samples is None else 'r'
        path = f'data/{config.name}/training/{base}-{dataset}.npy'

        if distributed:
            path = add_task_info(path)

        a = np.load(path, mode)
        if n_samples is not None:
            a = a[:n_samples].copy()

        return torch.as_tensor(a)
    
    u = load('u')
    rays = load('rays')
    Y = load(target_type)

    if target_type.startswith('flux'):
        _, grain = target_type.split('-')
        idx = {'fine' : [1, 0], 'coarse' : [1, 2]}[grain]
        u = u[:, idx]

        if kwargs.get('nondimensional', True):
            T = hp.generation.max_days * 86400
            k, *_, dk, dl, dm, dens = rays.T
            action = dens * dk * dl * dm

            factor = abs(k) * action * config.dr_init / T
            Y = Y / factor[:, None]

    elif target_type == 'adjustments':
        u = u[:, :-1]

    return u, rays, Y
