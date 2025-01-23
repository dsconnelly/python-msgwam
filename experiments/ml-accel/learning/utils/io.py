from __future__ import annotations
from typing import Optional

import numpy as np
import torch

from msgwam import config

from ... import hyperparameters as hp
from .distributed import add_task_info

def get_indices(
    eval_type: str,
    n_packets: Optional[int]=None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Get index tensors partitioning the data into training and evaluation sets.
    Note that no shuffling is performed, so that the training, validation, and
    test sets correspond to disjoint time periods within the integration.

    Parameters
    ----------
    eval_type
        Evaluation dataset specifier, as passed to `train_network`.
    n_packets
        How many packets the combined training, validation, and test sets should
        consist of. Useful for testing architectures on smaller datasets. If not
        provided, defaults to using all available packets.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Index tensors for the training and evaluation sets.

    """

    if n_packets is None:
        n_packets = hp.generation.n_packets

    a = int(0.7 * n_packets)
    b = int(0.85 * n_packets)
    idx = torch.arange(n_packets)

    idx_tr = idx[:a]
    idx_va = idx[a:b]
    idx_te = idx[b:]

    if eval_type == 'validation':
        return idx_tr, idx_va

    return torch.cat((idx_tr, idx_va)), idx_te

def load_data(
    target_type: str,
    distributed: bool=False,
    n_samples: Optional[int]=None,
    **kwargs
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load machine learning input and output data from disk.

    Parameters
    ----------
    target_type
        Kind of target data to load, as passed to `train_network`.
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

    def load(fname: str) -> torch.Tensor:
        mode = None if n_samples is None else 'r'
        path = f'data/{config.name}/training/{fname}'

        if distributed:
            path = add_task_info(path)

        a = np.load(path, mode)
        if n_samples is not None:
            a = a[:n_samples].copy()

        return torch.as_tensor(a)

    u = load('u.npy')
    rays = load('rays.npy')
    Y = load(f'{target_type}.npy')

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
