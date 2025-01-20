from __future__ import annotations
from time import time
from typing import TYPE_CHECKING, Callable, Optional
from warnings import catch_warnings

import torch, torch.nn as nn

from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from msgwam import config

from .. import hyperparameters as hp
from .architectures import Surrogate, get_model_dir, load_model, make_inputs
from .losses import FluxLoss
from .utils import (
    get_indices,
    get_overrides,
    load_data
)

if TYPE_CHECKING:
    from .architectures import SourceNet
    _TraceFunc = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

def train_network(
    grain: str='coarse',
    eval_type: str='validation',
    restart: bool=False,
    n_print: int=1
) -> None:
    """
    Wrapper around `_train_network` so that that function can be called with the
    appropriate override to `config.n_grid` and with sensible defaults. See that
    function's docstring for explanations of each argument.
    """

    hp.display()
    with config.override(n_grid=get_overrides()['n_grid']):
        _train_network(f'flux-{grain}', eval_type, restart, n_print)

def _train_network(
    target_type: str,
    eval_type: str,
    restart: bool,
    n_print: int
) -> None:
    """
    Train a `SourceNet` subclass. This function can be used either to train a
    network with particular hyperparameter settings as part of a grid search, or
    to retrain a network on the combined training and validation sets using the
    best hyperparameters found during tuning.

    Parameters
    ----------
    target_type
        What targets should be used, which implicity determines the `SourceNet`
        subclass to train. If training a `Surrogate`, must be of the form
        `'{kind}-{grain}'`, where `{kind}` is either `'flux'` or `'proxies`' and
        `{grain}` is either `'fine'` or `'coarse'`.
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
    loss_func = FluxLoss()

    print(model, '\n')
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model {hp.task_id} has {n_params} trainable parameters.\n')

    if not restart:
        u_tr, X_tr, _ = loader_tr.dataset.tensors
        model.init_stats(u_tr, X_tr)

    best_loss = torch.inf
    state = {'task_id' : hp.task_id}

    max_epochs = hp.training.max_epochs
    max_hours = hp.training.max_hours
    n_epoch, start = 1, time()

    while n_epoch <= max_epochs and (time() - start) / 3600 < max_hours:
        loss_tr = _run_epoch(model, loader_tr, loss_func, optimizer)
        # loss_tr = _run_epoch(model, loader_tr, loss_func)
        loss_ev = _run_epoch(model, loader_ev, loss_func)

        if n_epoch % n_print == 0:
            print(f'== epoch {n_epoch} == ')
            print(f'loss_tr = {loss_tr:.6f}')
            print(f'loss_ev = {loss_ev:.6f}')

        if loss_ev < best_loss:
            state['model'] = model.state_dict()
            state['optimizer'] = optimizer.state_dict()
            best_loss = loss_ev

        n_epoch = n_epoch + 1

    print(f'Best loss was {best_loss:.6f}')
    model.load_state_dict(state['model'])

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    trace_func = _make_trace_func(model)
    u_ex, rays_ex, _ = load_data(target_type)

    with torch.no_grad():
        with catch_warnings(action='ignore', category=torch.jit.TracerWarning):
            traced = torch.jit.trace(trace_func, (u_ex[:10], rays_ex[:10]))

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

    u, rays, targets = load_data(target_type)
    n_packets = min(hp.generation.n_packets, u.shape[0])
    idx_tr, idx_ev = get_indices(eval_type, n_packets)

    if target_type.startswith('flux'):
        targets = torch.clamp(abs(targets), max=1)
        targets = torch.cummin(targets, dim=1)[0]

    loaders = []
    for idx in (idx_tr, idx_ev):
        data = TensorDataset(*make_inputs(u[idx], rays[idx]), targets[idx])
        loaders.append(DataLoader(data, hp.training.batch_size, shuffle=True))

    return tuple(loaders)

def _make_trace_func(model) -> _TraceFunc:
    """
    Create a function that takes inputs as they will come during online use,
    evaluates the model, and postprocesses the outputs appropriately.

    Parameters
    ----------
    model
        Model whose behavior should be traced.

    Returns
    -------
    _TraceFunc
        Function to trace and save as a JITted object.

    """

    if isinstance(model, Surrogate):
        def trace_func(u: torch.Tensor, rays: torch.Tensor) -> torch.Tensor:
            """
            Stack the wind and ray volume information, extracting the spectral
            features as necessary. Then evaluate the model and make sure the
            sign of the returned flux is correct.
            """

            signs = torch.sign(rays[:, 0])[:, None]
            u, X = make_inputs(u, rays)
            Y = model(u, X)

            return signs * Y
        
        return trace_func

    return NotImplemented

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
        loss_func.eval()

        with torch.no_grad():
            u, X, targets = loader.dataset.tensors
            loss = loss_func(targets, model(u, X))

        return loss.item()

    model.train()
    loss_func.train()

    weight_sum, total = 0, 0
    for u, X, targets in loader:
        optimizer.zero_grad()
        weight = X.shape[0]

        c_noise = torch.normal(0, hp.training.noise_scale_c, X[:, 0].shape)
        u_noise = torch.normal(0, hp.training.noise_scale_u, u.shape)
        X[:, 0], u = X[:, 0] + c_noise, u + u_noise

        output = model(u, X)
        loss = loss_func(targets, output)
        total = total + weight * loss.item()
        weight_sum = weight_sum + weight

        loss.backward()
        optimizer.step()

    return total / weight_sum
