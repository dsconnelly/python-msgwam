from __future__ import annotations
from typing import Optional

import numpy as np
import torch

from msgwam import config

from .. import hyperparameters as hp

from .bases import apply_basis
from .overrides import get_overrides

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
        n_packets = hp.n_packets

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
    **kwargs
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load machine learning input and output data from disk.

    Parameters
    ----------
    target_type
        Kind of target data to load, as passed to `train_network`.
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

    data_dir = f'data/{config.name}/training'
    u = torch.as_tensor(np.load(f'{data_dir}/u.npy'))
    rays = torch.as_tensor(np.load(f'{data_dir}/rays.npy'))

    if target_type.startswith(('flux', 'proxies')):
        target_type, grain = target_type.split('-')

    if target_type == 'flux':
        Y = torch.as_tensor(np.load(f'{data_dir}/flux-{grain}.npy'))

        if kwargs.get('nondimensional', True):
            T = hp.max_days * 86400
            k, *_, dk, dl, dm, dens = rays.T
            action = dens * dk * dl * dm

            factor = abs(k) * action * config.dr_init / T
            Y = Y / factor[:, None]

    elif target_type == 'proxies':
        fname = f'proxies-{grain}-{hp.basis_type}.npy'
        Y = torch.as_tensor(np.load(f'{data_dir}/{fname}'))

        if kwargs.get('reconstructed', False):
            signs = torch.sign(rays[:, :1])
            signs = signs[:Y.shape[0]]

            n_grid = get_overrides()['n_grid']
            Y = signs * apply_basis(Y, n_grid)

    return u, rays, Y
