import os

from copy import deepcopy
from time import time
from typing import Iterator, Optional

import cftime
import numpy as np
import torch
import xarray as xr

from optuna import create_study, load_study
from optuna.exceptions import TrialPruned
from optuna.pruners import MedianPruner
from optuna.study import get_all_study_names
from optuna.trial import FixedTrial, Trial, TrialState

from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ReduceLROnPlateau,
    LRScheduler
)
from torch.utils.data import DataLoader, TensorDataset

from msgwam import config
from msgwam.constants import EPOCH
from msgwam.utils import gaussian_filter

from ... import hyperparameters as hp
from ...strategies import (
    get_integration,
    get_overrides
)

from ..architectures import ConvNet, UNet

from .inference import serialize_model
from .io import CMY, get_best_trial, prepare_data
from .losses import FluxLoss, VelocityLoss
from .transforms import Transform

_DEVICE = torch.device('cpu')
if torch.cuda.is_available():
    _DEVICE = torch.device('cuda')
    print('Training will occur on the GPU.')

torch.set_flush_denormal(True)

REFERENCES = {}

def search_hyperparameters(
    n_hours_small: str | int,
    n_hours_large: str | int
) -> None:
    """
    Search hyperparameter space for the best configuration.

    Parameters
    ----------
    n_hours_small
        How many hours to spend on the warmup study (using a small subset of the
        data to suggest promising candidates to the main study).
    n_hours_large
        How many hours to spend on the main study.
    
    """

    n_hours_small = int(n_hours_small)
    n_hours_large = int(n_hours_large)

    if hp.training.n_online_test > 0:
        _init_references()

    name = hp.training.exp_name
    path = f'sqlite:///data/ml-accel/models/study-{name}.db'
    n_warmup = max(hp.training.patience, hp.training.n_online_test)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=n_warmup)

    kwargs = {
        'gc_after_trial' : True,
        'catch' : (FloatingPointError, RuntimeError)
    }

    study_small = None
    info = get_all_study_names(path)

    if f'{name}-small' in info:
        print(f'Found pre-existing warmup study.')
        
        study_small = load_study(
            study_name=f'{name}-small',
            storage=path
        )

    elif n_hours_small > 0:
        study_small = create_study(
            direction='minimize',
            study_name=f'{name}-small',
            pruner=pruner,
            storage=path
        )

        study_small.optimize(
            func=(lambda t: _train(t, warmup=True)),
            timeout=(n_hours_small * 3600),
            **kwargs
        )

    if n_hours_large > 0:
        study_large = create_study(
            direction='minimize',
            study_name=f'{name}-large',
            load_if_exists=True,
            pruner=pruner,
            storage=path
        )

        if study_small is not None:
            keep = lambda t: t.state == TrialState.COMPLETE
            trials = [t for t in study_small.get_trials() if keep(t)]
            n_keep = min(5, max(1, int(0.2 * len(trials))))

            key = lambda t: t.value
            trials = sorted(trials, key=key)[:n_keep]
            trials = [t for t in trials if t.value < 0.5]
            print(f'Enqueuing {len(trials)} trials.\n')

            for trial in trials:
                study_large.enqueue_trial(
                    trial.params,
                    skip_if_exists=True
                )

        study_large.optimize(
            func=_train,
            timeout=(n_hours_large * 3600),
            **kwargs
        )

def train_network() -> None:
    """Train a network with the best set of hyperparameters."""

    _train(get_best_trial())

def _get_model(trial: Trial, n_bins: int, state: Optional[dict]=None) -> UNet:
    """
    Instantiate a model to train, potentially loading state from a previous run.

    Parameters
    ----------
    trial
        Current trial, used to define the model architecture.
    n_bins
        How many phase speed bins to use.
    state
        If not `None`, should be a dictionary with key `'model'` pointing to a
        state dictionary matching the current model architecture.

    Returns
    -------
    ConvNet
        Initialized model.
    
    """

    model = UNet(trial, n_bins)
    n_params = sum(param.numel() for param in model.parameters())
    print(f'Initialized model with {n_params} trainable parameters.')

    if n_params > 2000000:
        print('Model is too complex.')
        raise TrialPruned()

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

    optim_name = trial.suggest_categorical('optimizer', ['AdamW'])
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

def _get_scheduler(
    trial: Trial,
    optimizer: Optimizer,
    state: Optional[dict]=None
) -> Optional[LRScheduler]:
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
        'cosine' : CosineAnnealingLR
    }

    scheduler_name = trial.suggest_categorical('scheduler', schedulers.keys())
    kwargs = {}

    if scheduler_name == 'plateau':
        kwargs['factor'] = trial.suggest_float('plateau_factor', 0.1, 0.5)
        kwargs['patience'] = trial.suggest_int('plateau_patience', 3, 8)
        kwargs['mode'] = 'min'

    elif scheduler_name == 'cosine':
        kwargs['T_max'] = trial.suggest_int('cosine_T_max', 40, 80)
        kwargs['eta_min'] = trial.suggest_float(
            'cosine_eta_min',
            1e-7, 5e-6,
            log=True
        )

    scheduler = schedulers[scheduler_name](optimizer, **kwargs)

    if state is not None and 'scheduler' in state:
        scheduler.load_state_dict(state['scheduler'])
        print('Scheduler state loaded successfully.')

    return scheduler

def _iter_loaders(
    trial: Trial,
    tensors: CMY,
    idxs: tuple[np.ndarray, np.ndarray],
) -> Iterator[DataLoader]:
    """
    Package the training and evaluation data into `DataLoader` instances.

    Parameters
    ----------
    trial
        Current trial, used to set the batch size.
    arrays
        Reshaped and transformed inputs and outputs.
    idxs
        Index arrays separating the data into training and evaluation sets.

    Returns
    -------
    DataLoader, DataLoader
        Loaders for the training and evaluation sets.

    """

    total = 0
    for a in tensors:
        total = total + a.numel() * a.element_size()

    if total / (1024 ** 3) < 20:
        tensors = [a.to(_DEVICE) for a in tensors]
        print('Preloaded tensors to the training device.')

    batch_size = trial.suggest_int('batch_size', 256, 1024, step=128)
    batch_sizes = [batch_size, 4096]

    for i, (idx, n) in enumerate(zip(idxs, batch_sizes)):
        ds = TensorDataset(*[a[idx] for a in tensors])
        yield DataLoader(ds, n, i == 0)

def _init_references():
    """
    In hyperparameter search, each trial concludes by using the network in
    integrations of a few scenarios. This function populates a global dictionary
    of references with the data needed to perform that evaluation.
    """

    scenarios = [
        'lisbon-1',
        'maldives-3',
        'miami-10',
        'weddell-sea-7'
    ]

    for scenario in scenarios:
        path = f'data/ml-accel/integrations-1200/24/{scenario}.nc'
        with xr.open_dataset(path) as ds:
            F = ds['F_bulk'].sum('bin')
            lat = ds.attrs['latitude']

        units = f'seconds since {EPOCH}'
        F['time'] = cftime.num2date(F['time'].values, units)

        F = gaussian_filter(F, hours=24, z_faces=1500)
        F_x = F.isel(quadrant=0) - F.isel(quadrant=2)
        F_y = F.isel(quadrant=1) - F.isel(quadrant=3)

        REFERENCES[scenario] = (lat, F_x, F_y)

def _train(
    trial: Trial,
    n_print: int=1,
    restart: bool=False,
    warmup: bool=False
) -> float:
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

    name = hp.training.exp_name
    eval_type = 'te' if isinstance(trial, FixedTrial) else 'va'

    if eval_type == 'te':
        n_samples = None
    elif warmup:
        n_samples = 30000
    else:
        n_samples = 500000

    state = None
    if restart and eval_type == 'te':
        kwargs = dict(weights_only=True, map_location=torch.device('cpu'))
        state = torch.load(f'data/ml-accel/models/state-{name}.pkl', **kwargs)

    options = [1, 2, 3, 4, 5]
    i = trial.suggest_int('n_bin_idx', 1, len(options) - 1)
    n_bins = options[i]

    tensors, idxs, transforms = prepare_data(
        trial=trial,
        n_bins=n_bins,
        eval_type=eval_type,
        n_samples=n_samples,
        seed=n_bins
    )

    model = _get_model(trial, n_bins, state)
    optimizer = _get_optimizer(trial, model, state)
    scheduler = _get_scheduler(trial, optimizer, state)

    loader_tr, loader_ev = _iter_loaders(trial, tensors, idxs)
    loss_func = VelocityLoss(trial, loader_tr.dataset.tensors[-1])
    loss_func = loss_func.to(_DEVICE)

    state = {}
    best_score = torch.inf
    n_epoch, waited = 1, 0

    patience = hp.training.patience if eval_type == 'va' else 30
    max_epochs = hp.training.max_epochs if eval_type == 'va' else -180

    if warmup:
        max_epochs = int(1.5 * max_epochs)

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
        loss_bp = _run_epoch(model, loader_tr, loss_func, optimizer)
        loss_tr = _run_epoch(model, loader_tr, loss_func)
        loss_ev = _run_epoch(model, loader_ev, loss_func)
        runtime = time() - epoch_start

        if warmup:
            loss_ev = loss_tr

        improved = loss_ev < best_score - hp.training.min_delta
        suffix = ' (new best)' if improved else ''

        if n_epoch % n_print == 0:
            print(f'    ==== epoch {n_epoch} ({runtime:.2f} s) ====')
            print(f'      loss_bp = {loss_bp:.6f}')
            print(f'      loss_tr = {loss_tr:.6f}')
            print(f'      loss_ev = {loss_ev:.6f}{suffix}')

        if improved:
            state['model'] = deepcopy(model.state_dict())
            state['optimizer'] = deepcopy(optimizer.state_dict())

            if scheduler is not None:
                state['scheduler'] = deepcopy(scheduler.state_dict())

            best_score, waited = loss_ev, 0

        else:
            waited = waited + 1

            if waited == patience:
                print('Stopping early due to lack of improvement.')
                break

        should_prune = np.isnan(loss_ev)

        if hp.training.n_online_test == 0:
            trial.report(loss_ev, n_epoch)
            should_prune = should_prune or trial.should_prune()

        elif n_epoch % hp.training.n_online_test == 0:
            trial.report(_run_online_tests(model, transforms), n_epoch)
            should_prune = should_prune or trial.should_prune()

        if should_prune:
            raise TrialPruned()

        if scheduler is not None:
            needs_loss = isinstance(scheduler, ReduceLROnPlateau)
            scheduler.step(*([loss_ev] if needs_loss else []))
        
        n_epoch = n_epoch + 1

    if state:
        model.load_state_dict(state['model'])

    if eval_type == 'te':
        scripted = serialize_model(None, (model, *transforms))
        torch.jit.save(scripted, f'data/ml-accel/models/scripted-{name}.jit')
        torch.save(state, f'data/ml-accel/models/state-{name}.pkl')

    if hp.training.n_online_test > 0:
        return _run_online_tests(model, transforms)

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
        Nf, C, M, Y = [a.to(_DEVICE) for a in tensors]

        if optimizer is None:
            with torch.no_grad():
                Y_hat = model(Nf, C, M)

        else:
            optimizer.zero_grad()
            Y_hat = model(Nf, C, M)

        weight = M.shape[0]
        weight_sum = weight_sum + weight
        loss = loss_func(Nf, Y, Y_hat, reduce=True)
        total = total + weight * loss

        if optimizer is not None:
            loss.backward()
            optimizer.step()

    return (total / weight_sum).item()

def _run_online_tests(model: ConvNet, transforms: list[Transform]) -> float:
    """
    Integrate several scenarios using the trained network, and return the
    average flux RMSE, as a final evaluation step in hyperparameter search.

    Parameters
    ----------
    n_bins
        Number of bins in trained model.
    
    Returns
    -------
    float
        Normalized flux RMSE over all scenarios.

    """

    
    scripted = serialize_model(None, (model, *transforms))
    torch.jit.save(scripted, '.tmp-model.jit')

    kwargs = get_overrides('network', hp.training.exp_name, model._n_bins)
    kwargs['dt_output'] = hp.generation.dt_output
    kwargs['model_path'] = '.tmp-model.jit'

    score = 0
    for scenario, (lat, *refs) in REFERENCES.items():
        path = f'data/ml-accel/context/24/{scenario}.nc'
        kwargs['prescribed_mean_file'] = path
        kwargs['latitude'] = lat

        with config.override(**kwargs):
            ds = get_integration().isel(member=0)

        for ref, cs in zip(refs, ['ew', 'ns']):
            rms = np.sqrt((ref ** 2).mean('time'))
            keep = (20e3 <= ref['z_faces']) & (ref['z_faces'] <= 55e3)
            keep = keep.values

            pmf = ds[f'pmf_{cs[0]}'] + ds[f'pmf_{cs[1]}']
            rmse = np.sqrt(((pmf - ref) ** 2).mean('time'))

            rms = rms.isel(z_faces=keep)
            rmse = rmse.isel(z_faces=keep)
            score = score + (rmse / rms).mean('z_faces').item()

    model.to(_DEVICE)
    os.remove('.tmp-model.jit')

    return score / len(REFERENCES) / 2
