from time import time
from typing import Literal, Optional

import torch, torch.nn as nn

from torch.optim import Adam
from torch.utils.data import DataLoader

from ... import hyperparameters as hp

from ..architectures import BulkNet

from .io import get_best_task_id, get_loaders, get_split, load_tensors
from .losses import BulkLoss
from .transforms import get_shift_and_scale, transform

def train_network(
    eval_type: Literal['va', 'te'],
    state_path: Optional[str]=None
) -> None:
    """
    Train a neural network to advance the bulk momentum and velocity profiles.

    Parameters
    ----------
    eval_type
        Whether to use validation or test data as the evaluation set.

    """

    if eval_type == 'te':
        i = get_best_task_id()
        hp.load(hp.grid_path, i)
        print(f'Best hyperparameter setting was {i}.')

    torch.manual_seed(1234)
    hp.show_hyperparameters()

    M, cg, wind, targets = load_tensors(eval_type)
    idx_tr, idx_ev = get_split(M.shape[0], eval_type)

    M_stats = get_shift_and_scale(M[idx_tr], hp.training.in_transform)
    cg_stats = get_shift_and_scale(cg[idx_tr], hp.training.in_transform)
    wind_stats = get_shift_and_scale(wind[idx_tr], 'z')

    M = transform(M, *M_stats)
    cg = transform(cg, *cg_stats)
    wind = transform(wind, *wind_stats)

    args = [M, cg, wind, targets, idx_tr, idx_ev]
    loader_tr, loader_ev = get_loaders(*args)

    model = BulkNet()
    optimizer = Adam(model.parameters(), lr=hp.training.learning_rate)
    loss_func = BulkLoss(targets[idx_tr])

    if state_path is not None:
        state = torch.load(state_path, weights_only=True)
        print(f'Loading previous state from {state_path}')

        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])

    n_tr, n_ev = len(idx_tr), len(idx_ev)
    n_params = sum(p.numel() for p in model.parameters())
    
    word = {'va' : 'validation', 'te' : 'test'}[eval_type]
    print(f'Loaded {n_tr} training samples and {n_ev} {word} samples.')
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
    model.load_state_dict(state['model'])

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    del loader_tr, loader_ev
    M, cg, wind, _ = load_tensors('va')
    M, cg, wind = M[idx_tr[:10]], cg[idx_tr[:10]], wind[idx_tr[:10]]

    def trace_func(
        M: torch.Tensor,
        cg: torch.Tensor,
        wind: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Capture pipeline, excluding momentum budgeting (which can be done at
        integration time) but including input normalization, so that the various
        statistics arrays don't need to be saved separately.
        """

        M = transform(M, *M_stats)
        cg = transform(cg, *cg_stats)
        wind = transform(wind, *wind_stats)

        return model(M, cg, wind)
    
    with torch.no_grad():
        traced = torch.jit.trace(trace_func, (M, cg, wind))

    tag = 'best' if eval_type == 'te' else hp.task_id
    torch.save(state, f'data/ml-accel/models/state-{tag}.pkl')
    torch.jit.save(traced, f'data/ml-accel/models/model-{tag}.jit')

    with open(f'data/ml-accel/records/loss-{tag}.txt', 'w') as f:
        f.write(str(best_loss))

def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_func: BulkLoss,
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
        Module accepting `(M_hat, cg_hat, targets)` and computing the loss to
        use for gradient descent and model evaluation.
    optimizer
        Optimizer to use for gradient descent. If `None`, then this is an
        evaluation step and the weights are not changed.

    Returns
    -------
    float
        Average loss over all samples in the loader.

    """

    if optimizer is None:
        loss_func.eval()
    else:
        loss_func.train()

    model = model.eval() if optimizer is None else model.train()
    weight_sum, total = 0, 0

    for *inputs, targets in loader:
        if optimizer is None:
            with torch.no_grad():
                M_hat, cg_hat = model(*inputs)

        else:
            optimizer.zero_grad()
            M_hat, cg_hat = model(*inputs)

        weight = targets.shape[0]
        loss = loss_func(M_hat, cg_hat, targets)
        total = total + weight * loss.item()
        weight_sum = weight_sum + weight

        if optimizer is not None:
            loss.backward()
            optimizer.step()

    return total / weight_sum