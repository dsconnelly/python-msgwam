from time import time
from typing import Literal, Optional

import torch, torch.nn as nn

from torch.optim import Adam
from torch.utils.data import DataLoader

from .. import hyperparameters as hp

from .architectures import BulkNet
from .losses import BulkLoss
from .utils import get_loaders

def train_network(eval_type: Literal['va', 'te']) -> None:
    """
    Train a neural network to advance the bulk momentum and velocity profiles.

    Parameters
    ----------
    eval_type
        Whether to use validation or test data as the evaluation set.

    """

    torch.manual_seed(1234)
    hp.show_hyperparameters()

    model = BulkNet()
    optimizer = Adam(model.parameters(), lr=hp.training.learning_rate)
    loader_tr, loader_ev = get_loaders(eval_type)
    loss_func = BulkLoss(loader_tr)

    n_params = sum(p.numel() for p in model.parameters())
    print(f'Loaded model has {n_params} trainable parameters.')

    state = {}
    best_loss = torch.inf
    n_epoch, start = 1, time()

    max_epochs = hp.training.max_epochs
    max_hours = hp.training.max_hours

    while n_epoch <= max_epochs and (time() - start) / 3600 < max_hours:
        loss_tr = _run_epoch(model, loader_tr, loss_func, optimizer)
        loss_ev = _run_epoch(model, loader_ev, loss_func)
        
        print(f'==== epoch {n_epoch} ====')
        print(f'  loss_tr = {loss_tr:.6f}')
        print(f'  loss_ev = {loss_ev:.6f}')

        if loss_ev < best_loss:
            state['model'] = model.state_dict()
            state['optimizer'] = optimizer.state_dict()
            best_loss = loss_ev

        n_epoch = n_epoch + 1

    print(f'Best loss was {best_loss:.6f}')
    path = f'data/ml-accel/models/state-{hp.task_id}.pkl'
    torch.save(state, path)

def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_func: nn.Module,
    optimizer: Optional[Adam]=None
) -> float:
    """
    Run a training epoch, calculating the total loss over all batches in the
    provided loader. Works for both training and evaluation steps.

    Parameters
    ----------
    model
        Network to be trained.
    loader
        Loader containing training or evaluation samples.
    loss_func
        Module accepting `(M, cg, M_hat, cg_hat)` and computing the loss to use
        for gradient descent and model evaluation.
    optimizer
        Optimizer to use for gradient descent. If `None`, then this is an
        evaluation step and the weights are not changed.

    Returns
    -------
    float
        Average loss over all samples in the loader.

    """

    model = model.eval() if optimizer is None else model.train()
    weight_sum, total = 0, 0

    for *inputs, M_next, cg_next in loader:
        if optimizer is None:
            with torch.no_grad():
                M_hat, cg_hat = model(*inputs)

        else:
            optimizer.zero_grad()
            M_hat, cg_hat = model(*inputs)

        weight = M_next.shape[0]
        loss = loss_func(M_next, cg_next, M_hat, cg_hat)
        total = total + weight * loss.item()
        weight_sum = weight_sum + weight

        if optimizer is not None:
            loss.backward()
            optimizer.step()

    return total / weight_sum
