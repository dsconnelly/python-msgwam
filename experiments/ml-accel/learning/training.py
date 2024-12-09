from __future__ import annotations
from time import time
from typing import TYPE_CHECKING, Optional
from warnings import catch_warnings

import torch, torch.nn as nn

from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from . import hyperparameters as hp
from .utils import get_indices, get_model_dir, load_data, load_model

if TYPE_CHECKING:
    from .architectures import SourceNet

def train_network(
    target_type: str='fine',
    eval_type: str='validation',
    restart: bool=False,
    n_print: int=1
) -> None:
    """
    Train a `SourceNet` subclass. This function can be used either to train a
    network with particular hyperparameter settings as part of a grid search, or
    to retrain a network on the combined training and validation sets using the
    best hyperparameters found during tuning.

    Parameters
    ----------
    target_type
        What targets should be used, which implicitly determines the `SourceNet`
        subclass to train. Must be either `'coarse'` or `'fine'`.
    eval_type
        Whether to use `'validation'` data to evaluate and train only on the
        training data, or to hold out `'test'` data and train the model on the
        combined training and validation sets.
    restart
        Whether training should resume from a previously saved state.
    n_print
        Interval, in epochs, at which to print training and evaluation losses.

    """

    loader_tr, loader_ev = _load_datasets(target_type, eval_type)
    model, optimizer = load_model(target_type, eval_type, restart)
    loss_func = nn.MSELoss()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model {hp.task_id} has {n_params} trainable parameters.')

    if not restart:
        u_tr, X_tr, _ = loader_tr.dataset.tensors
        model.init_stats(u_tr, X_tr)

    n_epoch, start = 1, time()
    while n_epoch <= hp.max_epochs and (time() - start) / 3600 < hp.max_hours:
        loss_tr = _run_epoch(model, loader_tr, loss_func, optimizer)
        loss_ev = _run_epoch(model, loader_ev, loss_func)

        if n_epoch % n_print == 0:
            print(f'== epoch {n_epoch} == ')
            print(f'loss_tr = {loss_tr:.6f}')
            print(f'loss_ev = {loss_ev:.6f}')

        n_epoch = n_epoch + 1

    state = {
        'model' : model.state_dict(),
        'optimizer' : optimizer.state_dict(),
        'task_id' : hp.task_id
    }

    u_ex, X_ex, _ = loader_ev.dataset.tensors
    traced = _trace_model(u_ex, X_ex, model)

    model_dir = get_model_dir(target_type)
    tag = f'{"best" if eval_type == "test" else hp.task_id}'
    torch.jit.save(traced, f'{model_dir}/model-{tag}.jit')
    torch.save(state, f'{model_dir}/state-{tag}.pkl')

def _load_datasets(
    target_type: str,
    eval_type: str
) -> tuple[DataLoader, DataLoader]:
    """
    Load training and evaluation sets, wrapped in a `DataLoader`.

    Parameters
    ----------
    target_type
        Target specifier, as passed to `train_network`.
    eval_type
        Evaluation dataset specifier, as passed to `train_network`.

    Returns
    -------
    tuple[DataLoader, DataLoader]
        Triples of tensors (u, X, targets) for training and evaluation sets.

    """

    u, X, targets = load_data(target_type)
    idx_tr, idx_ev = get_indices(eval_type)

    loaders = []
    for idx in (idx_tr, idx_ev):
        data = TensorDataset(u[idx], X[idx], targets[idx])
        loaders.append(DataLoader(data, hp.batch_size, shuffle=True))

    return tuple(loaders)

def _run_epoch(
    model: SourceNet,
    loader: DataLoader,
    loss_func: nn.Module,
    optimizer: Optional[Adam]=None
) -> float:
    """
    Run one epoch of either training or evaluation.

    Parameters
    ----------
    model
        `SourceNet` instance being trained.
    loader
        `DataLoader` containing either the training or evaluation data.
    loss_func
        Module used to compute losses between targets and model outputs.
    optimizer
        If provided, optimizer associated to `model` that will be used to update
        the model weights. If `None`, it is assumed that this is the evaluation
        step, and weights are not updated.

    Returns
    -------
    float
        Average per-sample loss over the provided dataset.

    """

    if optimizer is None:
        model.eval()

        with torch.no_grad():
            u, X, targets = loader.dataset.tensors
            loss = loss_func(targets, model(u, X))

        return loss.item()

    model.train()
    weight_sum = 0
    total = 0

    for u, X, targets in loader:
        optimizer.zero_grad()
        output = model(u, X)
        weight = u.shape[0]

        loss = loss_func(targets, output)
        total = total + weight * loss.item()
        weight_sum = weight_sum + weight

        loss.backward()
        optimizer.step()

    return total / weight_sum

def _trace_model(
    u: torch.Tensor,
    X: torch.Tensor,
    model: SourceNet
) -> torch.jit.ScriptModule:
    """
    Trace a model with the JIT compiler.

    Parameters
    ----------
    u
        Example zonal wind profiles.
    X
        Example ray volume properties.
    model
        Trained model to be traced.

    Returns
    -------
    ScriptModule
        JITted model that can be saved to disk and subsequently called without
        needing access to the Python implementation of the class.

    """

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    with torch.no_grad():
        with catch_warnings(action='ignore', category=torch.jit.TracerWarning):
            return torch.jit.trace(model, (u, X))
