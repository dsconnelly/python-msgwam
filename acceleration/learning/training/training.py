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

    hp.show_hyperparameters()

    loader_tr, loader_ev, windN_stats, M_stats = _load_data(eval_type)
    loss_func = BulkLoss(loader_tr.dataset.tensors[-1])
    model, optimizer = _load_model(state_path)

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

    del loader_tr, loader_ev
    traced = _trace(model, windN_stats, M_stats)
    tag = 'best' if eval_type == 'te' else hp.task_id

    torch.save(state, f'data/ml-accel/models/state-{tag}.pkl')
    torch.jit.save(traced, f'data/ml-accel/models/model-{tag}.jit')

    with open(f'data/ml-accel/records/loss-{tag}.txt', 'w') as f:
        f.write(str(best_loss))

def _load_data(eval_type: Literal['va', 'te']) -> tuple[
    DataLoader,
    DataLoader,
    tuple[torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor]
]:
    """
    Load the data, partition it into training and evaluation sets, and transform
    it according to the loaded hyperparameters.

    Parameters
    ----------
    eval_type
        Evaluation type specifier, as passed to `train_network`.

    Returns
    -------
    DataLoader, DataLoader
        Loaders containing training and evaluation inputs and outpus.
    tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]
        Transforms for the mean state and bulk momentum budgets, respectively.
        These are returned so that they can be passed to `_trace` later.

    """

    windN, M, Y = load_tensors(eval_type)
    idx_tr, idx_ev = get_split(M.shape[0], eval_type)

    windN_stats = get_shift_and_scale(windN[idx_tr], 'z')
    M_stats = get_shift_and_scale(M[idx_tr], hp.training.in_transform)
    windN = transform(windN, *windN_stats)
    M = transform(M, *M_stats)

    n_tr, n_ev = len(idx_tr), len(idx_ev)
    word = {'va' : 'validation', 'te' : 'test'}[eval_type]
    max_res = abs(Y.sum(dim=1) - 1).max()

    print(f'Loaded {n_tr} training samples and {n_ev} {word} samples.')
    print(f'Maximum residual in targets is {max_res:.4e}.')

    loader_tr, loader_ev = get_loaders(windN, M, Y, idx_tr, idx_ev)
    return loader_tr, loader_ev, windN_stats, M_stats

def _load_model(state_path: Optional[str]=None) -> tuple[BulkNet, Adam]:
    """
    Load a `BulkNet` and associated optimizer, possibly loading state for both
    modules from a previous training run.

    Parameters
    ----------
    state_path
        Optional location of previous state for the model and optimizer.

    Returns
    -------
    BulkNet, Adam
        Model and optimizer ready for (further) training.

    """

    model = BulkNet()
    optimizer = Adam(model.parameters(), hp.training.learning_rate)
    n_params = sum(param.numel() for param in model.parameters())
    print(f'Loaded model has {n_params} trainable parameters.')

    if state_path is not None:
        state = torch.load(state_path, weights_only=True)
        print(f'Loading previous state from {state_path}.')

        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])

    return model, optimizer

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
        model.eval()
        loss_func.eval()

    else:
        model.train()
        loss_func.train()

    weight_sum, total = 0, 0
    for *inputs, Y in loader:
        if optimizer is None:
            with torch.no_grad():
                Y_hat = model(*inputs)

        else:
            optimizer.zero_grad()
            Y_hat = model(*inputs)

        weight = Y.shape[0]
        loss = loss_func(Y, Y_hat)
        total = total + weight * loss.item()
        weight_sum = weight_sum + weight

        if optimizer is not None:
            loss.backward()
            optimizer.step()

    return (total / weight_sum) ** 0.5

def _trace(
    model: BulkNet,
    windN_stats: tuple[torch.Tensor, torch.Tensor],
    M_stats: tuple[torch.Tensor, torch.Tensor]
) -> torch.jit.ScriptFunction:
    """
    Trace the model pipeline and return a JITted object that can be loaded by
    MS-GWaM without having to have the `BulkNet` class definition available.

    Parameters
    ----------
    model
        Trained `BulkNet`.
    windN_stats, M_stats
        Transform parameters that should be applied to the neural network inputs
        before passing them through the model.

    Returns
    -------
    ScriptFunction
        Traced model pipeline.

    """

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    windN, M, _ = load_tensors('va', min_samples=10)
    windN, M = windN[:10], M[:10]

    def trace_func(
        windN: torch.Tensor,
        M: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Capture pipeline, excluding momentum budgeting (which can be done at
        integration time) but including input normalization, so that the various
        statistics arrays don't need to be saved separately.
        """

        windN = transform(windN, *windN_stats)
        M = transform(M, *M_stats)

        return model(windN, M)

    with torch.no_grad():
        return torch.jit.trace(trace_func, (windN, M))