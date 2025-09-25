from typing import Literal

import torch, torch.nn as nn
import xarray as xr

from torch.utils.data import DataLoader, TensorDataset

from ..hyperparameters import training as hp

_SITES_TR = [
    'anchorage',
    'new-york',
    'lisbon',
    'miami',
    'maldives',
    'brisbane',
    'buenos-aires',
    'weddell-sea'
]

_SITES_TE = [
    'copenhagen',
    'singapore',
    'perth',
    'amundsen-sea'
]

def apply_blocks(blocks: nn.ModuleList, X: torch.Tensor) -> torch.Tensor:
    """
    Apply a set of blocks with skip connections after all but the last.

    Parameters
    ----------
    blocks
        List of modules to apply between skip connections.
    X
        Tensor to pass through the blocks.

    Returns
    -------
    torch.Tensor
        Output of final block.

    """

    output = X
    for block in blocks[:-1]:
        output = X + block(output)

    return blocks[-1](output)

def get_loaders(
    eval_type: Literal['va', 'te'],
) -> tuple[DataLoader, DataLoader]:
    """
    Load training and evaluation data from disk.

    Parameters
    ----------
    eval_type
        String specifying whether the evaluation data should be from the
        validation or test set.

    Returns
    -------
    DataLoader, DataLoader
        Loaders containing training and evaluation data. If `eval_type == 'va'`,
        the training and evaluation data are both sampled from `_SITES_TR`. If
        `eval_type == 'te'`, the training data are from `_SITES_TR` while the
        evaluation data are from `_SITES_TE`.

    """

    args = [[], [], [], [], [], []]
    for site in _SITES_TR + _SITES_TE * (eval_type == 'te'):
        with xr.open_dataset(f'data/ml-accel/training/mima-{site}.nc') as ds:
            u = torch.as_tensor(ds['u'].values)
            v = torch.as_tensor(ds['v'].values)

            M = torch.as_tensor(ds['M_bulk'].values)
            cg = torch.as_tensor(ds['cg_bulk'].values)
            source = torch.as_tensor(ds['source'].values)

        u, v = u[:, None], v[:, None]
        quad = torch.arange(4)[None, :, None]
        u, v, quad = torch.broadcast_tensors(u, v, quad)

        is_zonal = (torch.remainder(quad, 2) == 0).int()
        wind = is_zonal * u + (1 - is_zonal) * v
        wind[quad > 1] = -wind[quad > 1]

        args[0].append(M[:-1].flatten(0, 1))
        args[1].append(cg[:-1].flatten(0, 1))
        args[2].append(source[:-1].flatten(0, 1)[:, None])
        args[3].append(wind[:-1].flatten(0, 1))

        args[4].append(M[1:].flatten(0, 1))
        args[5].append(cg[1:].flatten(0, 1))

    Xs = [torch.vstack(arg) for arg in args]
    n_samples = Xs[0].shape[0]

    if eval_type == 'va':
        c = int(0.8 * n_samples)
        idx = torch.randperm(n_samples)

    else:
        c = (n_samples * len(_SITES_TR)) // len(_SITES_TR + _SITES_TE)
        idx = torch.arange(n_samples)

    ds_tr = TensorDataset(*[X[idx[:c]] for X in Xs])
    ds_ev = TensorDataset(*[X[idx[c:]] for X in Xs])
    loader_tr = DataLoader(ds_tr, hp.batch_size, shuffle=True)
    loader_ev = DataLoader(ds_ev, 1024, shuffle=False)

    return loader_tr, loader_ev

def xavier_init(layer: nn.Module) -> None:
    """
    Apply Xavier initialization a linear layer.

    Parameters
    ----------
    a
        Module to potentially initialize. If an `nn.Linear` is passed, its
        weight matrix will be initialized.

    """

    if isinstance(layer, nn.Linear):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(layer.weight, gain=gain)
