from __future__ import annotations
from os import listdir
from typing import TYPE_CHECKING, Any, Literal, Optional

import numpy as np
import torch, torch.nn as nn

from torch.optim import Adam

from msgwam import config

from . import architectures
from . import hyperparameters as hp

if TYPE_CHECKING:
    from .architectures import SourceNet

def apply_basis(
    coeffs: torch.Tensor,
    n_grid: Optional[int]=None
) -> torch.Tensor:
    """
    Given a tensor of amplitude, shape, and shift parameters, compute the
    profile given as the sum of the basis functions for each sample.

    Parameters
    ----------
    coeffs
        Tensor whose first dimension ranges over training samples and whose
        second dimension ranges over coefficients for the basis functions.

    Returns
    -------
    torch.Tensor
        Profile corresponding to each sample.

    """

    if n_grid == None:
        n_grid = config.n_grid

    n_samples = coeffs.shape[0]
    coeffs = coeffs.reshape(n_samples, 3, -1, 1)
    amp, shape, shift = coeffs.transpose(0, 1)
    z = -torch.linspace(-3, 3, n_grid)

    amp = torch.softmax(amp, dim=1)
    shape = nn.functional.softplus(shape)
    shift = 1.1 * z.max() * torch.tanh(shift)

    arg = shape * (z - shift)
    curves = amp * _basis_func(arg)

    return curves.sum(dim=1)

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

def get_overrides(fine: bool=False) -> dict[str, Any]:
    """
    Get the configuration overrides for various machine learning training steps.
    
    Parameters
    ----------
    fine
        Whether to get overrides for the reference fine integration or, if
        `False`, for the coarse integration.

    Returns
    -------
    dict[str, Any]
        Dictionary to pass to `config.override`.

    """

    root = int(hp.speedup ** 0.5)
    mean_path = f'data/{config.name}/input/descending-jets-training.nc'
    spectrum_path = f'data/{config.name}/input/spectrum-training.nc'

    n_day = _get_n_day()
    dt_output = n_day * 86400

    kwargs = {
        'source_type' : 'packet',
        'prescribed_wind_file' : mean_path,
        'spectrum_file' : spectrum_path,
        'dt' : 30,
        'dt_output' : dt_output,
        'n_day' : n_day,
        'n_grid' : 101,
        'dt_launch' : hp.dt_launch,
        'max_age' : hp.max_days * 86400,
        'n_increment' : 1000,
        'prune_by' : 'none',
    }

    if not fine:
        kwargs['n_chromatic'] = 1
        kwargs['n_repeat'] = 1

        return kwargs

    kwargs['n_source'] = config.n_source * root
    kwargs['dr_init'] = config.dr_init / root
    kwargs['n_chromatic'] = hp.speedup
    kwargs['n_repeat'] = root

    return kwargs

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

            Y = signs * apply_basis(Y, get_overrides()['n_grid'])
            
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
    
def _basis_func(z: torch.Tensor) -> torch.Tensor:
    """
    Compute the normalized version of the basis function, which must have
    unit slope at the origin and be bounded between zero and one.

    Parameters
    ----------
    z
        Tensor of input values.

    Returns
    -------
    torch.Tensor
        Basis function values.

    """

    if hp.basis_type == 'logistic':
        return 1 / (1 + torch.exp(-4 * z))
    
    if hp.basis_type == 'quadratic':
        return (1 + 2 * z / torch.sqrt(1 + (2 * z) ** 2)) / 2
    
    if hp.basis_type == 'tanh':
        return (1 + torch.tanh(2 * z)) / 2

    raise ValueError(f'Unknown basis type: {hp.basis_type}')

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

def _get_n_day() -> int:
    """
    Get a reasonable upper bound on the number of days the integration should
    run to generate the number of packets requested.

    Returns
    -------
    int
        Number of days to be prepared to integrate for.

    """

    last_launch = hp.dt_launch * hp.n_packets / config.n_source
    return int(last_launch / 86400 + hp.max_days + 5)