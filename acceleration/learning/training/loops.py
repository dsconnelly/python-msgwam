from __future__ import annotations
from collections import OrderedDict
from time import time
from typing import TYPE_CHECKING, Any, Callable, Optional

import torch, torch.nn as nn

from torch.optim import Adam
from torch.utils.data import DataLoader

from msgwam import config

from ... import hyperparameters as hp
from ..architectures import BaseNet

from .io import get_loader
from .utils import get_kinds, get_pipeline_spec, get_subsets, standardize

if TYPE_CHECKING:
    Transform = Callable[[torch.Tensor], torch.Tensor]

def train_networks(phase: str, eval_type: str) -> None:
    """
    Train the neural networks associated with a particular phase.

    Parameters
    ----------
    phase
        Which phase of training to run. Can be `'encoding'`, `'stepping'`, or
        `'joint'`, in which case all the networks will be tuned together.
    eval_type
        Whether to use `'validation'` or `'test'` sets for evaluation.

    """

    spec = get_pipeline_spec(phase)
    subsets = get_subsets(eval_type)
    tag = 'best' if eval_type == 'test' else str(hp.task_id)

    args = OrderedDict()
    for name, mode in spec.items():
        kwargs = {'name' : name, 'tag' : None if mode == 'new' else tag}
        model = BaseNet.from_kwargs(**kwargs)

        if mode == 'frozen':
            for p in model.parameters():
                p.requires_grad_(False)

        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'{name.capitalize()} has {n} trainable parameters')
        args[name] = model

    kinds = get_kinds(phase)
    subsets = get_subsets(eval_type)
    loader_tr = get_loader(kinds, subsets[0])
    loader_ev = get_loader(kinds, subsets[1])

    if phase == 'stepping':
        transform = args['encoder']

    else:
        loader = get_loader('F', subsets[0], 4096)
        fluxes = torch.vstack([a[0] for a in loader])
        means, stds = fluxes.mean(dim=0), fluxes.std(dim=0)
        transform = lambda a: standardize(a, means, stds)[0]

        torch.save(means, f'data/{config.name}/models/means-{tag}.pkl')
        torch.save(stds, f'data/{config.name}/models/stds-{tag}.pkl')

    model = nn.Sequential(args)
    state = _train(model, loader_tr, loader_ev, transform)

    for name, mode in spec.items():
        if mode == 'frozen':
            continue

        parse = lambda k: '.'.join(k.split('.')[1:])
        _state = {parse(k) : v for k, v in state.items() if k.startswith(name)}
        torch.save(_state, f'data/{config.name}/models/{name}-{tag}.pkl')

def _train(
    model: nn.Module,
    loader_tr: DataLoader,
    loader_ev: DataLoader,
    transform: Transform,
    n_print: int=1
) -> dict[str, Any]:
    """
    Run the main training loop for an initialized model.

    Parameters
    ----------
    model
        Model to train.
    loader_tr, loader_ev
        Loaders for training and evaluation sets, respectively.
    transform
        Transform to apply to targets before computing losses.
    n_print
        How often to print training and evaluation scores.

    Returns
    -------
    dict[str, Any]
        Trained model state.

    """

    best_loss = torch.inf
    state = {'task_id' : hp.task_id}

    def loss_func(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the MSE between outputs and transformed targets."""

        return ((output - transform(target)) ** 2).mean()

    lr = hp.training.learning_rate
    decay = hp.training.weight_decay * lr
    optimizer = Adam(model.parameters(), lr, weight_decay=decay)

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

        if True or loss_ev < best_loss:
            state['model'] = model.state_dict()
            state['optimizer'] = optimizer.state_dict()
            best_loss = loss_ev

        n_epoch = n_epoch + 1

    print(f'Best loss was {best_loss:.6f}')
    model.load_state_dict(state['model'])

    return state['model']

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
        loss = loss_func(output, Y)
        total = total + weight * loss.item()
        weight_sum = weight_sum + weight

        if optimizer is not None:
            loss.backward()
            optimizer.step()

    return total / weight_sum