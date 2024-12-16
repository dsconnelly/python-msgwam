from __future__ import annotations
from typing import TYPE_CHECKING, Optional
from os import listdir

import numpy as np
import torch

from msgwam import config

from .. import architectures
from .. import hyperparameters as hp

from .bases import apply_basis
from .overrides import get_overrides

if TYPE_CHECKING:
    from torch.optim import Adam
    from ..architectures import SourceNet

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

def get_model_dir(target_type: str) -> str:
    """
    Return the path to the directory containing models with the given targets.

    Parameters
    ----------
    target_type
        Target specifier, as passed to `train_network`.

    Returns
    -------
    str
        Path to the appropriate directory.

    """

    cls_name = _get_class_name(target_type).lower()
    return f'data/{config.name}/{cls_name}-{target_type}'

def load_data(
    target_type: str,
    grain: str,
    **kwargs
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load machine learning input and output data from disk.

    Parameters
    ----------
    target_type
        Kind of target data to load. Must be either `'flux'` or `'coeffs'`.
    grain
        Whether to load data corresponding to `'coarse'` or `'fine'` packets.
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

    if target_type == 'flux':
        Y = torch.as_tensor(np.load(f'{data_dir}/flux-{grain}.npy'))

        if kwargs.get('nondimensional', True):
            T = hp.max_days * 86400
            k, *_, dk, dl, dm, dens = rays.T
            action = dens * dk * dl * dm

            factor = abs(k) * action * config.dr_init / T
            Y = Y / factor[:, None]

    elif target_type == 'coeffs':
        fname = f'coeffs-{grain}-{hp.basis_type}.npy'
        Y = torch.as_tensor(np.load(f'{data_dir}/{fname}'))

        if kwargs.get('reconstructed', False):
            signs = torch.sign(rays[:, :1])
            signs = signs[:Y.shape[0]]

            n_grid = get_overrides()['n_grid']
            Y = signs * apply_basis(Y, n_grid)
            
    return u, rays, Y

def load_model(
    target_type: str,
    eval_type: str,
    restart: bool
) -> tuple[SourceNet, Adam]:
    """
    Load a model and an associated optimizer. Can be used to initialize a new
    model or to load a trained model from disk. If loading a model trained on
    the full training and validation sets, the `task_id` will be changed so that
    the saved weights are compatible with the model structure.

    Parameters
    ----------
    target_type
        Target specifier, as passed to `train_network`.
    eval_type
        Evaluation dataset specifier, as passed to `train_network`. If `test`,
        the hyperparameter `task_id` may be modified.
    restart
        Whether to load saved state from disk.

    Returns
    -------
    SourceNet
        Requested subclass instance, with loaded state if necessary.
    Adam
        Associated optimizer, with loaded state if necessary.

    """

    if restart:
        model_dir = get_model_dir(target_type)
        tag = 'best' if eval_type == 'test' else hp.task_id
        state = torch.load(f'{model_dir}/state-{tag}.pkl')

        if eval_type == 'test':
            hp.load(hp.grid_path, state['task_id'])

    elif eval_type == 'test':
        task_id = _get_best_task_id()
        print(f'Selecting hyperparameter configuration {task_id}:')
        hp.load(hp.grid_path, _get_best_task_id(), verbose=True)
        print()

    cls_name = _get_class_name(target_type)
    model: SourceNet = getattr(architectures, cls_name.capitalize())()
    optimizer = Adam(model.parameters(), hp.learning_rate)

    if restart:
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])

    return model, optimizer

def _get_best_task_id() -> int:
    """
    Get the task ID of the training run with the lowest validation score by
    reading the log files.

    Returns
    -------
    int
        Task ID of the best hyperparameter configuration.

    """

    best_id = None
    best_score = np.inf

    log_dir = 'logs/ml-accel'
    for fname in listdir(log_dir):
        if not fname.startswith('training-'):
            continue

        with open(f'{log_dir}/{fname}') as f:
            for line in f:
                if not line.startswith('loss_ev'):
                    continue

                score = float(line.strip().split(' = ')[1])

        if score < best_score:
            best_id = int(fname.split('.')[0].split('-')[-1])
            best_score = score

    return best_id

def _get_class_name(target_type: str) -> str:
    """
    Return the name of the `SourceNet` subclass that can be trained to learn
    targets with the given specifier.

    Parameters
    ----------
    target_type
        Target specifier, as passed to `train_network`.

    Returns
    -------
    str
        Name of the appropriate `SourceNet` subclass.

    """

    return {'coarse' : 'Surrogate', 'fine' : 'Surrogate'}[target_type]
