from time import time
from typing import Literal, Optional

import json

import torch, torch.nn as nn

from optuna import create_study
from optuna.exceptions import TrialPruned
from optuna.pruners import MedianPruner
from optuna.trial import FixedTrial, Trial

from torch.optim import Adam
from torch.utils.data import DataLoader

from msgwam import config

from ... import hyperparameters as hp

from ..architectures import BulkNet

from .io import get_loaders, get_split, load_tensors
from .losses import BulkLoss
from .transforms import get_shift_and_scale, transform

_DEVICE = torch.device('cpu')
_CACHED = True

def search_hyperparameters() -> None:
    """
    Conduct a hyperparameter search to determine the best network architecture
    and training scheme. The training data is loaded in advance, but reshaping
    is deferred until the model is constructed and the number of bins is known.
    """

    _set_device()
    *inputs, M, D = load_tensors('va', cached=_CACHED)
    objective = lambda t: _train(t, *inputs, M, D)

    n_samples = M.shape[0]
    budget = M.sum(dim=(1, 2)) + D.sum(dim=1)
    residual = abs(1 - budget).max().item()

    print(f'Loaded {n_samples} total samples.')
    print(f'Max residual is {residual:.4e}')

    pruner = MedianPruner(5, 10)
    study = create_study(direction='minimize', pruner=pruner)
    study.optimize(objective, timeout=(2 * 3600), gc_after_trial=True)
    trial = study.best_trial

    print('==== Summary ====')
    print(f'  best score: {trial.value:.4f}')
    for key, value in trial.params.items():
        print(f'  {key}: {value}')

    with open(f'data/ml-accel/models/hyperparameters-best.json', 'w') as f:
        json.dump(trial.params, f)

def train_network() -> None:
    """
    Train a model on the best hyperparameter set. Must be called after
    `search_hyperparameters` has already been run.
    """

    _set_device()
    with open('data/ml-accel/models/hyperparameters-best.json') as f:
        trial = FixedTrial(json.load(f), -1)

    _train(trial, *load_tensors('te', cached=_CACHED))

def _train(
    trial: Trial,
    *datas: torch.Tensor
) -> float:
    """
    Train a network with a given `Trial` and return the best evaluation loss.

    Parameters
    ----------
    trial
        Trial to use to generate relevant hyperparameters.
    datas
        Training and evaluation inputs and targets.

    Returns
    -------
    float
        Best evaluation loss over all epochs.

    """

    eval_type = 'te' if trial.number == -1 else 'va'
    model, optimizer = _load_model(trial)

    args = (model._n_bins, trial, eval_type, *datas)
    loader_tr, loader_ev, windN_stats, M_stats = _prepare_data(*args)
    loss_func = BulkLoss(*loader_tr.dataset.tensors[-2:]).to(_DEVICE)

    state = {}
    best_loss = torch.inf
    n_epoch, start = 1, time()
    waited = 0

    max_epochs = hp.training.max_epochs
    max_hours = hp.training.max_hours
    min_delta = hp.training.min_delta

    while n_epoch <= max_epochs and (time() - start) / 3600 < max_hours:
        epoch_start = time()
        loss_tr = _run_epoch(model, loader_tr, loss_func, optimizer)
        loss_ev = _run_epoch(model, loader_ev, loss_func)
        runtime = time() - epoch_start

        print(f'  ==== epoch {n_epoch} ({runtime:.2f} s) ====')
        print(f'    loss_tr = {loss_tr:.6f}')
        print(f'    loss_ev = {loss_ev:.6f}')

        trial.report(loss_ev, n_epoch)
        if trial.should_prune():
            raise TrialPruned()

        if loss_ev < best_loss - min_delta:
            state['model'] = model.state_dict()
            state['optimizer'] = optimizer.state_dict()

            best_loss = loss_ev
            waited = 0

        else:
            waited = waited + 1
            if waited == hp.training.patience:
                print(f'Stopping early.')
                break

        n_epoch = n_epoch + 1

    if trial.number == -1:
        del loader_tr, loader_ev
        traced = _trace(model, windN_stats, M_stats)

        torch.save(state, f'data/ml-accel/models/state-best.pkl')
        torch.jit.save(traced, 'data/ml-accel/models/model-best.jit')

    return best_loss

def _load_model(trial: Trial) -> tuple[BulkNet, Adam]:
    """
    Load a `BulkNet` and associated optimizer.

    Parameters
    ----------
    trial
        Current trial from which to draw a learning rate.

    Returns
    -------
    BulkNet, Adam
        Model and optimizer ready for (further) training.

    """

    model = BulkNet(trial).to(_DEVICE)
    lr = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    optimizer = Adam(model.parameters(), lr=lr)

    if n_params > hp.architectures.max_params:
        raise TrialPruned()

    n_params = sum(param.numel() for param in model.parameters())
    print(f'Loaded model has {n_params} trainable parameters.')

    return model, optimizer

def _prepare_data(
    n_bins: int,
    trial: Trial,
    eval_type: Literal['va', 'te'],
    *args: torch.Tensor
) -> tuple[
    DataLoader,
    DataLoader,
    tuple[torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor]
]:
    """
    Prepare the already-loaded data for a given trial. Reshapes the momentum
    profiles to include the appropriate number of bins, transforms the input
    features, and packages the tensors in `DataLoader` instances.
    """

    windN, M_in, M_out, D = args
    shape = (M_in.shape[0], n_bins, -1, config.n_grid - 1)
    M_in, M_out = [a.reshape(*shape).sum(dim=2) for a in (M_in, M_out)]

    idx_tr, idx_ev = get_split(M_in.shape[0], eval_type)
    windN_stats = get_shift_and_scale(windN[idx_tr], 'z')
    M_stats = get_shift_and_scale(M_in[idx_tr], hp.training.in_transform)

    windN = transform(windN, *windN_stats)
    M_in = transform(M_in, *M_stats)

    batch_size = trial.suggest_int('batch_size', 128, 512)
    args = [a.to(_DEVICE) for a in (windN, M_in, M_out, D)]
    loader_tr, loader_ev = get_loaders(batch_size, idx_tr, idx_ev, *args)

    return loader_tr, loader_ev, windN_stats, M_stats

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
    for *inputs, M, D in loader:
        if optimizer is None:
            with torch.no_grad():
                M_hat, D_hat = model(*inputs)

        else:
            optimizer.zero_grad()
            M_hat, D_hat = model(*inputs)

        weight = M.shape[0]
        loss = loss_func(M, D, M_hat, D_hat)
        weight_sum = weight_sum + weight
        total = total + weight * loss

        if optimizer is not None:
            loss.backward()
            optimizer.step()

    return (total / weight_sum).item() ** 0.5

def _set_device() -> None:
    """Set the global `_DEVICE` depending on whether a GPU is available."""

    global _DEVICE
    if torch.cuda.is_available():
        _DEVICE = torch.device('cuda')
        print('Training will occur on the GPU.')

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
    M = M.reshape(M.shape[0], model._n_bins, -1, config.n_grid - 1)
    windN, M = windN[:10], M.sum(dim=2)[:10]

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