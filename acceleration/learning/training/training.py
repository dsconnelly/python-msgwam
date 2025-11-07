import json

from copy import deepcopy
from time import time
from typing import Iterator, Optional

import numpy as np
import torch

from optuna import create_study
from optuna.exceptions import TrialPruned
from optuna.pruners import MedianPruner
from optuna.trial import FixedTrial, Trial

from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    ReduceLROnPlateau,
    LRScheduler
)
from torch.utils.data import DataLoader, TensorDataset

from ... import hyperparameters as hp

from ..architectures import ConvNet

from .inference import serialize_model
from .io import CMY, prepare_data
from .losses import FluxLoss

_DEVICE = torch.device('cpu')
if torch.cuda.is_available():
    _DEVICE = torch.device('cuda')
    print('Training will occur on the GPU.')

torch.set_flush_denormal(True)

_CACHE = {}

def search_hyperparameters(n_hours_str: str) -> None:
    """
    Search hyperparameter space for the best configuration.

    Parameters
    ----------
    n_hours_str
        How many hours to conduct the search for, passed as a string so that
        this function may be invoked from the command line.
    
    """

    study = create_study(
        direction='minimize',
        study_name='msgwam-convnet',
        pruner=MedianPruner(5, hp.training.patience),
        storage='sqlite:///data/ml-accel/models/study.db',
        load_if_exists=True
    )

    n_hours = float(n_hours_str)
    kwargs = {'gc_after_trial' : True, 'catch' : (RuntimeError,)}
    study.optimize(_train, timeout=(n_hours * 3600), **kwargs)

    params = study.best_trial.params
    with open('data/ml-accel/models/hyperparameters.json', 'w') as f:
        json.dump(params, f, indent=4)

def train_network() -> None:
    """Train a network with the best set of hyperparameters."""

    with open('data/ml-accel/models/hyperparameters.json') as f:
        _train(FixedTrial(json.load(f), number=-1))

def _get_model(trial: Trial, state: Optional[dict]=None) -> ConvNet:
    """
    Instantiate a model to train, potentially loading state from a previous run.

    Parameters
    ----------
    trial
        Current trial, used to define the model architecture.
    state
        If not `None`, should be a dictionary with key `'model'` pointing to a
        state dictionary matching the current model architecture.

    Returns
    -------
    ConvNet
        Initialized model.
    
    """

    model = ConvNet(trial)
    n_params = sum(param.numel() for param in model.parameters())
    print(f'Initialized model with {n_params} trainable parameters.')

    if state is not None:
        model.load_state_dict(state['model'])
        print('Previous model state loaded successfullly.')

    return model.to(_DEVICE)

def _get_optimizer(
    trial: Trial,
    model: ConvNet,
    state: Optional[dict]=None
) -> tuple[Optimizer]:
    """
    Initialize an optimizer to use during training.

    Parameters
    ----------
    trial
        Current trial, used to select and configure the optimizer.
    model
        Model to train.
    state
        If not `None`, should contain a key `'optimizer'` pointing to a state
        dictionary matching the current optimizer class and configuration.

    Returns
    -------
    Optimizer
        Optimizer bound to model to be trained.

    """

    optim_name = trial.suggest_categorical('optimizer', ['AdamW', 'SGD'])
    lr_bounds = {'AdamW' : (1e-5, 5e-3), 'SGD' : (1e-2, 5e-1)}[optim_name]
    lr = trial.suggest_float('learning_rate', *lr_bounds, log=True)
    kwargs = {'lr' : lr}

    if optim_name == 'SGD':
        momentum = trial.suggest_float('momentum', 0.8, 0.99)
        kwargs['momentum'] = momentum
        kwargs['nesterov'] = True

    if trial.suggest_categorical('use_wd', [True, False]):
        wd_bounds = {'AdamW' : (1e-3, 1e-1), 'SGD' : (1e-5, 1e-1)}[optim_name]
        weight_decay = trial.suggest_float('weight_decay', *wd_bounds, log=True)
        kwargs['weight_decay'] = weight_decay

    optim_cls = getattr(torch.optim, optim_name)
    optimizer = optim_cls(model.parameters(), **kwargs)

    if state is not None:
        optimizer.load_state_dict(state['optimizer'])
        print(f'Previous optimizer state loaded successfully.')

    return optimizer

def _get_scheduler(trial: Trial, optimizer: Optimizer) -> Optional[LRScheduler]:
    """
    Create a learning rate scheduler for the optimizer.

    Parameters
    ----------
    trial
        Current trial, used to select and configure the scheduler.
    optimizer
        Configured optimizer to be used during training.

    Returns
    -------
    Optional[LRScheduler]
        Scheduler to use to update the learning rate, or `None` if no scheduler
        is to be used this trial.
    
    """

    schedulers = {
        'none' : lambda *_: None,
        'plateau' : ReduceLROnPlateau,
        'cosine' : CosineAnnealingWarmRestarts
    }

    scheduler_name = trial.suggest_categorical('scheduler', schedulers.keys())
    kwargs = {}

    if scheduler_name == 'plateau':
        kwargs['factor'] = trial.suggest_float('plateau_factor', 0.1, 0.5)
        kwargs['patience'] = trial.suggest_int('plateau_patience', 3, 8)
        kwargs['mode'] = 'min'

    elif scheduler_name == 'cosine':
        kwargs['T_0'] = trial.suggest_int('cosine_T_0', 10, 30)
        kwargs['T_mult'] = trial.suggest_int('cosine_T_mult', 1, 3)

    return schedulers[scheduler_name](optimizer, **kwargs)

def _iter_loaders(
    tensors: CMY,
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

    Returns
    -------
    DataLoader, DataLoader
        Loaders for the training and evaluation sets.

    """

    batch_sizes = [hp.training.batch_size, 4096]
    for i, (idx, batch_size) in enumerate(zip(idxs, batch_sizes)):
        ds = TensorDataset(*[a[idx] for a in tensors])
        yield DataLoader(ds, batch_size, i == 0)

def _train(trial: Trial, n_print: int=1, restart: bool=False) -> float:
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
    n_print
        How frequently to print loss reports.

    Returns
    -------
    float
        Best evaluation loss over all epochs.
    
    """

    eval_type = 'te' if isinstance(trial, FixedTrial) else 'va'
    n_samples = 500000 if eval_type == 'va' else None

    state = None
    if restart and eval_type == 'te':
        kwargs = dict(weights_only=True, map_location=torch.device('cpu'))
        state = torch.load('data/ml-accel/models/state-best.pkl', **kwargs)

    model = _get_model(trial, state)
    optimizer = _get_optimizer(trial, model, state)
    scheduler = _get_scheduler(trial, optimizer)

    try:
        tensors, idxs, transforms = _CACHE[model._n_bins]
        print(f'Found data for {model._n_bins} bins in the cache.')

    except KeyError:
        tensors, idxs, transforms = prepare_data(
            trial=trial,
            n_bins=model._n_bins,
            eval_type=eval_type,
            n_samples=n_samples,
            seed=model._n_bins
        )

        if eval_type == 'va':
            _CACHE[model._n_bins] = (tensors, idxs, transforms)

    loader_tr, loader_ev = _iter_loaders(tensors, idxs)
    loss_func = FluxLoss(loader_tr.dataset.tensors[-1])
    loss_func = loss_func.to(_DEVICE)

    state = {}
    best_score = torch.inf
    n_epoch, waited = 1, 0

    patience = hp.training.patience if eval_type == 'va' else -1
    max_epochs = hp.training.max_epochs if eval_type == 'va' else -60

    if max_epochs > 0:
        keep_going = lambda n, _: n <= max_epochs

    else:
        start = time()
        max_minutes = abs(max_epochs)
        keep_going = lambda _, t: (t - start) / 60 < max_minutes

    units = 'epochs' if max_epochs > 0 else 'minutes'
    print(f'Training will continue for {abs(max_epochs)} {units}.\n')

    while keep_going(n_epoch, time()):
        epoch_start = time()
        loss_tr = _run_epoch(model, loader_tr, loss_func, optimizer)
        loss_ev = _run_epoch(model, loader_ev, loss_func)
        runtime = time() - epoch_start

        improved = loss_ev < best_score - hp.training.min_delta
        suffix = ' (new best)' if improved else ''

        if n_epoch % n_print == 0:
            print(f'    ==== epoch {n_epoch} ({runtime:.2f} s) ====')
            print(f'      loss_tr = {loss_tr:.6f}')
            print(f'      loss_ev = {loss_ev:.6f}{suffix}')

        if np.isnan(loss_ev):
            print('NaN detected with parameters')
            for key, value in trial.params.items():
                print(f'    {key}: {value}')

            raise TrialPruned()

        trial.report(loss_ev, n_epoch)
        too_slow = (eval_type == 'va') and (n_epoch == 2) and (runtime > 45)
        if trial.should_prune() or too_slow:
            raise TrialPruned()

        if improved:
            state['model'] = deepcopy(model.state_dict())
            state['optimizer'] = deepcopy(optimizer.state_dict())
            best_score, waited = loss_ev, 0

        else:
            waited = waited + 1

            if waited == patience:
                print('Stopping early due to lack of improvement.')
                break

        if scheduler is not None:
            needs_loss = isinstance(scheduler, ReduceLROnPlateau)
            scheduler.step(*([loss_ev] if needs_loss else []))
        
        n_epoch = n_epoch + 1

    if eval_type == 'te':
        model.load_state_dict(state['model'])
        serialize_model((model, *transforms))
        torch.save(state, 'data/ml-accel/models/state-best.pkl')

    return best_score

def _run_epoch(
    model: ConvNet,
    loader: DataLoader,
    loss_func: FluxLoss,
    optimizer: Optional[torch.optim.Optimizer]=None
) -> list[float]:
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
    list[float]
        Losses for `Y`, `W`, and aggregated.

    """

    if optimizer is None:
        model.eval()
        loss_func.eval()

    else:
        model.train()
        loss_func.train()

    weight_sum = 0
    total = 0

    for tensors in loader:
        C, M, *targets = [a.to(_DEVICE) for a in tensors]

        if optimizer is None:
            with torch.no_grad():
                outputs = model(C, M)

        else:
            optimizer.zero_grad()
            outputs = model(C, M)

        weight = M.shape[0]
        weight_sum = weight_sum + weight
        loss = loss_func(*targets, outputs)
        total = total + weight * loss

        if optimizer is not None:
            loss.backward()
            optimizer.step()

    p = {'mse' : 2, 'smae' : 1}[hp.training.loss_func]
    return (total / weight_sum).item() ** (1 / p)
