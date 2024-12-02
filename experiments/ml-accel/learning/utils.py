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

def load_model(target_type: str, restart: int) -> tuple[SourceNet, Adam]:
    """
    Load a model of the specified kind and an associated optimizer. If this is a
    restart run, load the saved state for both modules.

    Parameters
    ----------
    target_type
        Target specifier, as passed to `train_network`.
    restart
        What training restart this is.
    
    Returns
    -------
    SourceNet
        Requested subclass instance, with loaded state if necessary.
    Adam
        Associated optimizer, with loaded state if necessary.

    """

    cls_name = {'coarse' : 'Surrogate', 'fine' : 'Surrogate'}[target_type]
    model: SourceNet = getattr(architectures, cls_name.capitalize())()
    optimizer = Adam(model.parameters(), hp.learning_rate)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model {hp.task_id} has {n_params} trainable parameters.')

    if restart > 0:
        fname = f'state-{hp.task_id}-r{restart - 1}.pkl'
        state = torch.load(f'data/{config.name}/models/{fname}')

        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        print(f'Loaded state from training run {restart - 1}')

    return model, optimizer
