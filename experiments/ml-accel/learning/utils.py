from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
import torch

from torch.optim import Adam

from msgwam import config

from . import architectures
from . import hyperparameters as hp

if TYPE_CHECKING:
    from .architectures import SourceNet

def get_indices(eval_type: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Get index tensors partitioning the data into training and evaluation sets.
    Note that no shuffling is performed, so that the training, validation, and
    test sets correspond to disjoint time periods within the integration.

    Parameters
    ----------
    eval_type
        Evaluation dataset specifier, as passed to `train_network`.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Index tensors for the training and evaluation sets.

    """

    a = int(0.7 * hp.n_packets)
    b = int(0.85 * hp.n_packets)
    idx = torch.arange(hp.n_packets)
    
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
    nondimensional: bool=True
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load machine learning input and output data from disk.

    Parameters
    ----------
    target_type
        What targets to load. Must be either `'coarse'` or `'fine'`.
    nondimensional
        If the targets are flux profiles, whether to nondimensionalize them
        before returning.

    Returns
    -------
    torch.Tensor
        Loaded zonal wind profiles.
    torch.Tensor
        Loaded ray volume properties.
    torch.Tensor
        Loaded targets.

    """

    u = np.load(f'data/{config.name}/u.npy')
    X = np.load(f'data/{config.name}/X.npy')
    Y = np.load(f'data/{config.name}/Y-{target_type}.npy')

    if nondimensional:
        T = hp.max_days * 86400
        k, *_, dk, dl, dm, dens = X.T
        action = dens * dk * dl * dm

        factor = abs(k) * action * config.dr_init / T
        Y = Y / factor[:, None]

    return torch.as_tensor(u), torch.as_tensor(X), torch.as_tensor(Y)

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

    cls_name = _get_class_name(target_type)
    model: SourceNet = getattr(architectures, cls_name.capitalize())()
    optimizer = Adam(model.parameters(), hp.learning_rate)

    if restart:
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])

    return model, optimizer
    
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