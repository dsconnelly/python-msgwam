import numpy as np
import torch

from msgwam import config

from . import hyperparameters as hp

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
