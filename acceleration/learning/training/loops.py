from __future__ import annotations
from collections import OrderedDict
from time import time
from typing import TYPE_CHECKING, Any, Callable, Optional

import torch, torch.nn as nn

from torch.optim import Adam
from torch.utils.data import DataLoader

from msgwam import config

from ... import hyperparameters as hp
from .. import architectures

from .io import _Phase, get_loader
from .utils import get_flux_statistics, standardize

if TYPE_CHECKING:
    from ..architectures import BaseNet

    Transform = Callable[[torch.Tensor], torch.Tensor]

def train_network(phase: _Phase, eval_type: str) -> None:
    """
    
    """

    phase = int(phase)

    if phase == 1:
        pipeline = {
            'encoder' : 'new',
            'observer' : 'new'
        }

    elif phase == 2:
        pipeline = {
            'encoder' : 'frozen',
            'stepper' : 'new'
        }

    elif phase == 3:
        pipeline = {
            'encoder' : 'load',
            'stepper' : 'load',
            'observer' : 'load'
        }

    tag = 'best' if eval_type == 'test' else str(hp.task_id)
    loader_tr, loader_ev = _load_datasets(phase, eval_type)
    args = OrderedDict()

    for name, mode in pipeline.items():
        model: BaseNet = getattr(architectures, name.capitalize())()

        if mode in ['load', 'frozen']:
            path = f'data/{config.name}/models/{name}-{tag}.pkl'
            model.load_state_dict(torch.load(path))

        if mode == 'frozen':
            for p in model.parameters():
                p.requires_grad_(False)

        args[name] = model

    if phase == 2:
        transform = args['encoder']

    else:
        means, stds = get_flux_statistics(loader_tr.dataset)
        transform = lambda a: standardize(a, means, stds)[0]

    model = nn.Sequential(args)
    states = _train(model, loader_tr, loader_ev, transform)
    states = states['model']

    torch.save(means, f'data/{config.name}/training/flux-means.pkl')
    torch.save(stds, f'data/{config.name}/training/flux-stds.pkl')

    for name, mode in pipeline.items():
        if mode == 'frozen':
            continue
        
        state = {}
        for k, v in states.items():
            if not k.startswith(name):
                continue

            state['.'.join(k.split('.')[1:])] = v

        path = f'data/{config.name}/models/{name}-{tag}.pkl'
        torch.save(state, path)

def _load_datasets(
    phase: _Phase,
    eval_type: str
) -> tuple[DataLoader, DataLoader]:
    """
    Load training and evaluation sets, wrapped in `DataLoader` objects.

    Parameters
    ----------
    phase
        Phase of training, to be passed to `MultifileDataset`.
    eval_type
        Evaluation dataset specifier, as passed to `train_network`.

    Returns
    -------
    tuple[DataLoader, DataLoader]
        Tuples of input and target tensors for the training and evaluation sets.

    """

    if eval_type == 'validation':
        subsets_tr = ['tr']
        subsets_ev = ['va']

    elif eval_type == 'test':
        subsets_tr = ['tr', 'va']
        subsets_ev = ['te']

    else:
        raise ValueError(f'Unknown eval_type: {eval_type}')
    
    loaders = []
    for subsets in (subsets_tr, subsets_ev):
        loaders.append(get_loader(phase, subsets))

    return tuple(loaders)

def _train(
    model: nn.Module,
    loader_tr: DataLoader,
    loader_ev: DataLoader,
    transform: Transform,
    n_print: int=1
) -> dict[str, Any]:
    """
    
    """

    best_loss = torch.inf
    state = {'task_id' : hp.task_id}

    def loss_func(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        
        """

        return ((output - transform(target)) ** 2).mean()

    optimizer = Adam(
        model.parameters(),
        lr=hp.training.learning_rate,
        weight_decay=(hp.training.weight_decay * hp.training.learning_rate)
    )

    max_epochs = hp.training.max_epochs
    max_hours = hp.training.max_hours
    n_epoch, start = 1, time()

    while n_epoch <= max_epochs and (time() - start) / 3600 < max_hours:
        loss_tr = _run_epoch(model, loader_tr, loss_func, optimizer)
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

    return state

def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_func: nn.Module,
    optimizer: Optional[Adam]=None
) -> float:
    """
    Run one epoch of either training or evaluation.

    Parameters
    ----------
    model
        Model instance being trained
    loader
        `DataLoader` containing either training or evaluation data.
    loss_func
        Module used to compute losses between targets and model outputs.
    optimizer
        If provided, optimizer associated to `model` that will be used to update
        the model weights. If `None`, it is assumed that this is an evaluation
        step, and weights are not updated.

    Returns
    -------
    float
        Average per-sample loss over the provided dataset.

    """

    if optimizer is None:
        model.eval()

    else:
        model.train()

    weight_sum = 0
    total = 0

    for *Xs, Y in loader:
        if optimizer is None:
            with torch.no_grad():
                output = model(*Xs)

        else:
            optimizer.zero_grad()
            output = model(*Xs)

        weight = Y.shape[0]
        loss = loss_func(Y, output)
        total = total + weight * loss.item()
        weight_sum = weight_sum + weight

        if optimizer is not None:
            loss.backward()
            optimizer.step()

    return total / weight_sum