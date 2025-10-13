import json

from copy import deepcopy
from time import time
from typing import Iterator, Literal, Optional

import numpy as np
import torch

from optuna import create_study
from optuna.exceptions import TrialPruned
from optuna.pruners import MedianPruner
from optuna.trial import FixedTrial, Trial

from torch.utils.data import DataLoader, TensorDataset

from ... import hyperparameters as hp

from ..architectures import BulkNet

from .io import CMYW, parse_integrations, prepare_data, trace
from .losses import BulkLoss

_DEVICE = torch.device('cpu')
if torch.cuda.is_available():
    _DEVICE = torch.device('cuda')
    print('Training will occur on the GPU.')

def cache_arrays() -> None:
    """
    Convenience function to cache training arrays with the current settings, to
    be called e.g. before submitting a job to a GPU node.
    """

    _ = parse_integrations(cached=False)

def search_hyperparameters() -> None:
    """Search hyperparameter space for the best set."""

    arrays = parse_integrations(cached=True)
    objective = lambda t: _train(t, arrays)

    pruner = MedianPruner(5, hp.training.min_epochs)
    study = create_study(direction='minimize', pruner=pruner)
    study.optimize(objective, timeout=(20 * 60), gc_after_trial=True)
    trial = study.best_trial

    with open('data/ml-accel/models/hyperparameters.json', 'w') as f:
        json.dump(trial.params, f, indent=4)

def train_network() -> None:
    """Train a network with the best set of hyperparameters."""

    with open('data/ml-accel/models/hyperparameters.json') as f:
        trial = FixedTrial(json.load(f))

    _train(trial, parse_integrations(cached=True))

def _get_model(
    trial: Trial,
    eval_type: Literal['va', 'te'],
    state_path: Optional[str]=None
) -> tuple[BulkNet, torch.optim.Adam]:
    """
    Instantiate a `BulkNet` and an associated optimizer, possibly loading
    state from a previous training run.

    Parameters
    ----------
    trial
        Current trial, used to instantiate the neural network architecture and
        to sample a learning rate.
    eval_type
        Evaluation data type specifier, used to set `model._relax_beta`.
    state_path
        If provided, a path to existing model state. Will only work if `trial`
        is a `FixedTrial`. If not provided, a new model is returned.

    Returns
    -------
    BulkNet, Adam
        Model and optimizer ready for training.

    """

    model = BulkNet(trial, eval_type == 'te')
    lr = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    n_params = sum(param.numel() for param in model.parameters())
    print(f'Loaded model has {n_params} trainable parameters.')

    if n_params > hp.architectures.max_params:
        raise TrialPruned()
    
    if state_path is not None:
        print(f'Loading previous state from {state_path}')
        state = torch.load(state_path, weights_only=True)

        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])

    return model.to(_DEVICE), optimizer

def _iter_loaders(
    trial: Trial,
    arrays: CMYW,
    idxs: tuple[np.ndarray, np.ndarray],
) -> Iterator[DataLoader]:
    """
    Package the training and evaluation data into `DataLoader` instances.

    Parameters
    ----------
    arrays
        Reshaped and transformed inputs and outputs.
    idxs
        Index arrays separating the data into training and evaluation sets.
    batch_size_tr
        Batch size to use for the training data.
        
    Returns
    -------
    DataLoader, DataLoader
        Loaders for the training and evaluation sets.

    """

    batch_sizes = [trial.suggest_int('batch_size', 64, 1024), 4096]
    tensors = [torch.as_tensor(a).to(_DEVICE) for a in arrays]

    for i, (idx, batch_size) in enumerate(zip(idxs, batch_sizes)):
        ds = TensorDataset(*[a[idx] for a in tensors])
        yield DataLoader(ds, batch_size, i == 0)

def _train(trial: Trial, arrays: CMYW) -> float:
    """
    Train a network with the given `Trial` and return the best evaluation loss.
    It is assumed that the data has been read in from the netCDF files already,
    but not reshaped or transformed.

    Parameters
    ----------
    trial
        Object from which to sample relevant hyperparameters. If it is a
        `FixedTrial`, it is assumed that we are training the best network after
        hyperparameter search, and the resulting network is saved. Otherwise, it
        is assumed that we are in hyperparameter search.
    arrays
        Tuple of unprocessed input and output data.

    Returns
    -------
    float
        Best evaluation loss over all epochs.
    
    """

    eval_type = 'te' if isinstance(trial, FixedTrial) else 'va'
    model, optimizer = _get_model(trial, eval_type)

    arrays, idxs, transforms = prepare_data(model._n_bins, eval_type, arrays)
    loader_tr, loader_ev = _iter_loaders(trial, arrays, idxs)
    loss_func = BulkLoss(*loader_tr.dataset.tensors[-2:])
    loss_func = loss_func.to(_DEVICE)

    state = {}
    best_score = torch.inf
    n_epoch, waited = 1, 0

    max_epochs = hp.training.max_epochs
    max_epochs = max_epochs * (1 + (eval_type == 'te'))

    while n_epoch <= max_epochs:
        epoch_start = time()
        losses_tr = _run_epoch(model, loader_tr, loss_func, optimizer)
        losses_ev = _run_epoch(model, loader_ev, loss_func)
        runtime = time() - epoch_start

        print(f'    ==== epoch {n_epoch} ({runtime:.2f} s) ====')
        for losses, suffix in zip([losses_tr, losses_ev], ['tr', 'ev']):
            print(f'      loss_{suffix}  = {losses[2]:.6f}')
            print(f'        loss_Y = {losses[0]:.6f}')
            print(f'        loss_W = {losses[1]:.6f}')

        (*_, loss_tr), (*_, loss_ev) = losses_tr, losses_ev
        score = (1 - model.progress) * loss_tr + model.progress * loss_ev
        improved = score < best_score - hp.training.min_delta
        
        suffix = ' (new best)' if improved else ''
        print(f'      score    = {score:.6f}{suffix}')

        trial.report(loss_ev, n_epoch)
        if trial.should_prune():
            raise TrialPruned()

        if improved:
            state['model'] = deepcopy(model.state_dict())
            state['optimizer'] = deepcopy(optimizer.state_dict())
            best_score, waited = score, 0

        elif n_epoch > hp.training.min_epochs - hp.training.patience:
            waited = waited + 1

            if waited == hp.training.patience:
                print('Stopping early due to lack of improvement.')
                break

        model.update_beta(loss_tr)
        n_epoch = n_epoch + 1

    if eval_type == 'te':
        del loader_tr, loader_ev
        model.load_state_dict(state['model'])
        traced = trace(model, *transforms)

        torch.save(state, 'data/ml-accel/models/state-best.pkl')
        torch.jit.save(traced, 'data/ml-accel/models/model-best.jit')

    return best_score

def _run_epoch(
    model: BulkNet,
    loader: DataLoader,
    loss_func: BulkLoss,
    optimizer: Optional[torch.optim.Adam]=None
) -> tuple[float, float, float]:
    """
    Run a training or evaluation epoch, calculating the total loss over all
    batches in the provided loader.

    Parameters
    ----------
    model
        Network to be trained.
    loader
        Loader containing training or evaluation samples.
    loss_func
        Module to compute the appropriate loss function.
    optimizer
        Optimizer to use for gradient descent. If `None`, then this is an
        evaluation step and the weights are not updated.

    Returns
    -------
    float
        Root-mean-square loss over all samples in the loader.

    """

    if optimizer is None:
        model.eval()
        loss_func.eval()

    else:
        model.train()
        loss_func.train()

    weight_sum = 0
    total_Y, total_W, total = 0, 0, 0
    
    for *inputs, Y, W in loader:
        if optimizer is None:
            with torch.no_grad():
                Y_hat, W_hat = model(*inputs)

        else:
            optimizer.zero_grad()
            Y_hat, W_hat = model(*inputs)

        loss_Y, loss_W = loss_func(Y, W, Y_hat, W_hat)
        loss = loss_Y + loss_W

        weight = Y.shape[0]
        weight_sum = weight_sum + weight

        total_Y = total_Y + weight * loss_Y
        total_W = total_W + weight * loss_W
        total = total + weight * loss

        if optimizer is not None:
            loss.backward()
            optimizer.step()

    totals = [total_Y, total_W, total]
    rms = lambda a: (a / weight_sum).item() ** 0.5

    return tuple(map(rms, totals))
