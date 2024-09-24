from __future__ import annotations
from time import time
from typing import TYPE_CHECKING
from warnings import catch_warnings

import torch

from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader, TensorDataset

import architectures
import hyperparameters as params

from losses import ProfileLoss
from utils import DATA_DIR, nondimensionalize

if TYPE_CHECKING:
    from architectures import SourceNet

SURROGATE_TAG = '62-r0'

def train_network(
    kind: str,
    max_epochs: int=50,
    max_hours: int=2,
    restart: int=0,
    n_print: int=1
) -> None:
    """
    
    """

    coarse = kind == 'surrogate'
    loader_tr, loader_va, _ = _load_data(coarse)
    model, optimizer = _load_model(kind, restart)
    loss_func = _get_loss_func(kind)

    if restart == 0:
        u_tr, Y_tr, _ = loader_tr.dataset.tensors
        model.init_stats(u_tr, Y_tr)

    n_epoch, start = 1, time()
    while n_epoch <= max_epochs and (time() - start) / 3600 < max_hours:
        loss_tr = _train_epoch(model, loader_tr, optimizer, loss_func)
        loss_va = _validate_epoch(model, loader_va, loss_func)

        if n_epoch % n_print == 0:
            print(f'\n==== epoch {n_epoch} ====')
            print(f'loss_tr = {loss_tr:.8f}')
            print(f'loss_va = {loss_va:.8f}')

        n_epoch = n_epoch + 1

    state = {
        'model' : model.state_dict(),
        'optimizer' : optimizer.state_dict()
    }

    u_ex, Y_ex, _ = loader_va.dataset.tensors
    traced = _trace_model(u_ex, Y_ex, model)

    tag = f'{params.task_id}-r{restart}'
    torch.save(state, f'{DATA_DIR}/{kind}/state-{tag}.pkl')
    torch.jit.save(traced, f'{DATA_DIR}/{kind}/model-{tag}.jit')

def _load_data(coarse: bool) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load training, validation, and test sets, as defined by the indices saved in
    `DATA_DIR`. For convenience, each subset is returned in a `DataLoader`.

    Returns
    -------
    tuple[DataLoader, DataLoader, DataLoader]
        Triples of tensors (u, Y, Z) for training, validation, and test sets.

    """

    suffix = '-coarse' if coarse else ''
    u = torch.load(f'{DATA_DIR}/wind.pkl')[:, 0, 0]
    Y = torch.load(f'{DATA_DIR}/Y.pkl').transpose(1, 2)
    Z = torch.load(f'{DATA_DIR}/Z{suffix}.pkl')

    u = u[:, None].expand(*Z[..., :-1].shape).flatten(0, 1)
    Y, Z = Y.flatten(0, 1), Z.flatten(0, 1)
    Z = nondimensionalize(Y, Z)

    idx_tr = torch.load(f'{DATA_DIR}/idx-tr.pkl')
    idx_va = torch.load(f'{DATA_DIR}/idx-va.pkl')
    idx_te = torch.load(f'{DATA_DIR}/idx-te.pkl')

    print('Loaded ' + (
        f'{len(idx_tr)} training samples, ' +
        f'{len(idx_va)} validation samples, ' +
        f'and {len(idx_te)} test samples'
    ))

    loaders = []
    for idx in (idx_tr, idx_va, idx_te):
        data = TensorDataset(u[idx], Y[idx], Z[idx])
        loaders.append(DataLoader(data, params.batch_size, shuffle=True))

    return tuple(loaders)

def _get_loss_func(kind: str) -> ProfileLoss:
    """
    Create a loss function, including a surrogate if necessary.

    Parameters
    ----------
    kind
        Whether to train a `'surrogate'` or `'tinkerer'` model.
    
    Returns
    -------
    ProfileLoss
        Initialized loss function, with a surrogate if `kind == 'tinkerer'`.

    """

    surrogate = None
    if kind == 'tinkerer':
        path = f'{DATA_DIR}/surrogate/model-{SURROGATE_TAG}.jit'
        surrogate = torch.jit.load(path)

    return ProfileLoss(surrogate)

def _load_model(kind: str, restart: int) -> tuple[SourceNet, Optimizer]:
    """
    Load a model of the specified kind and an associated optimizer, and load any
    saved state for both modules if this is a restart run.

    Parameters
    ----------
    kind
        Whether to train a `'surrogate'` or `'tinkerer'` model.
    restart
        What training restart this is. Zero corresponds to a new model.

    Returns
    -------
    SourceNet
        Requested subclass instance, with loaded state if necessary.
    Optimizer
        Associated optimizer, with loaded state if necessary.

    """

    cls_name = kind.capitalize()
    model: SourceNet = getattr(architectures, cls_name)()
    optimizer = Adam(model.parameters(), params.learning_rate)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'model {params.task_id} has {n_params} trainable parameters')

    if restart > 0:
        fname = f'state-{params.task_id}-r{restart - 1}.pkl'
        state = torch.load(f'{DATA_DIR}/{kind}/{fname}')

        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        print(f'Loaded state from training run {restart - 1}')

    return model, optimizer

def _trace_model(
    u: torch.Tensor,
    Y: torch.Tensor,
    model: SourceNet,
) -> torch.jit.ScriptModule:
    """
    Trace a model so that it can be saved as a JITted function.

    Parameters
    ----------
    u
        Two-dimensional tensor of example zonal wind profiles.
    Y
        Two-dimensional tensor of example coarse ray volume properties.
    model
        Trained model to trace.

    Returns
    -------
    ScriptModule
        JITted model that can be saved to disc and called without access to the
        Python implementation of the class.

    """

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    with torch.no_grad():
        with catch_warnings(action='ignore', category=torch.jit.TracerWarning):
            traced = torch.jit.trace(model, (u, Y))

    return traced

def _train_epoch(
    model: SourceNet,
    loader: DataLoader,
    optimizer: Optimizer,
    loss_func: ProfileLoss
) -> float:
    """
    Run one epoch of model training.

    Parameters
    ----------
    model
        `SourceNet` subclass instance to be trained.
    loader
        Loader containing (u, Y, Z) training triples.
    optimizer
        Optimizer updating the parameters of `model`.
    loss_func
        `ProfileLoss` initialized correctly with respect to `model`.

    Returns
    -------
    float
        Loss averaged over all batches in the training epoch. Returned as a
        float, detached from the graph and suitable for printing.

    """

    model.train()
    weight_sum = 0
    total = 0

    for u, Y, Z in loader:
        optimizer.zero_grad()
        output = model(u, Y)
        weight = u.shape[0]

        loss = loss_func(u, output, Z)
        total = total + weight * loss.item()
        weight_sum = weight_sum + weight

        loss.backward()
        optimizer.step()

    return total / weight_sum

def _validate_epoch(
    model: SourceNet,
    loader: DataLoader,
    loss_func: ProfileLoss
) -> float:
    """
    Evaluate the model on the validation set.

    Parameters
    ----------
    model
        `SourceNet` subclass instance to evaluate.
    loader
        Loader containing (u, Y, Z) validation triples.
    loss_func
        `ProfileLoss` initialized correctly with respect to `model`.

    Parameters
    ----------
    float
        Loss averaged over all samples in the validation set.

    """

    model.eval()
    with torch.no_grad():
        u, Y, Z = loader.dataset.tensors
        loss = loss_func(u, model(u, Y), Z)

    return loss.item()