from __future__ import annotations
from os import listdir
from time import time
from warnings import catch_warnings

import torch, torch.nn as nn

from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

import architectures
import hyperparameters as params

from architectures import SourceNet, Surrogate, Tinkerer
from utils import DATA_DIR, LOG_DIR, load_data

def train_network(
    cls_name: str,
    target_type: str,
    eval_data: str='validation',
    restart: int=0,
    max_epochs: int=500,
    max_hours: int=2,
    n_print: int=1 
) -> None:
    """
    Train a `SourceNet` subclass. This function can be used either to train a
    network with particular hyperparameter settings part of a grid search, or to
    retrain a network on the training and validation sets using the best
    hyperparameters found during tuning.

    Parameters
    ----------
    cls_name
        What kind of network to train. Must be the name of a `SourceNet`
        subclass defined in `architectures`, lowercase.
    target_type
        What kind of targets should be used. Must be used to specify `'fine'` or
        `'coarse'` flux profiles for `Surrogate` models, and shoudl be set to
        `'Y_hat'` for `Tinkerer` models.
    eval_data
        Whether to use `'validation'` data to evaluate and train only on the
        training data, or to hold out `'test'` data and train the model on the
        combined training and validation sets.
    restart
        Whether the training run should resume from a previously saved state.
        If nonzero, state must be saved for run `restart - 1`.
    max_epochs
        How many epochs to perform, time permitting.
    max_hours
        How long to allow training to proceed before interrupting and saving.
    n_print
        Interval, in epochs, at which to print training and validation losses.

    """

    if eval_data == 'test':
        k = _get_best_task_id(cls_name, target_type)
        params.load(params.grid_path, k)

    loss_func = nn.MSELoss()
    model, optimizer = _load_model(cls_name, restart)
    loader_tr, loader_ev = _load_datasets(target_type, eval_data)

    if restart == 0:
        u_tr, Y_tr, _ = loader_tr.dataset.tensors
        model.init_stats(u_tr, Y_tr)

    n_epoch, start = 1, time()
    while n_epoch <= max_epochs and (time() - start) / 3600 < max_hours:
        loss_tr = _train_epoch(loader_tr, model, optimizer, loss_func)
        loss_ev = _evaluate_epoch(loader_ev, model, loss_func)

        if n_epoch % n_print == 0:
            print(f'\n==== epoch {n_epoch} ====')
            print(f'loss_tr = {loss_tr:.6f}')
            print(f'loss_ev = {loss_ev:.6f}')

        n_epoch = n_epoch + 1

    state = {
        'model' : model.state_dict(),
        'optimizer' : optimizer.state_dict()
    }

    u_ex, Y_ex, _ = loader_ev.dataset.tensors
    traced = _trace_model(u_ex, Y_ex, model)

    folder = cls_name + '-' + target_type
    tag = f'{"best" if eval_data == "test" else params.task_id}-r{restart}'
    torch.jit.save(traced, f'{DATA_DIR}/{folder}/model-{tag}.jit')
    torch.save(state, f'{DATA_DIR}/{folder}/state-{tag}.pkl')

def _get_best_task_id(cls_name: str, target_type: str) -> int:
    """
    Read the log files and find the task id (index into the hyperparameter grid)
    that achieved the lowest validation error.

    Parameters
    ----------
    cls_name
        `SourceNet` subclass being trained, as passed to `train_network`.
    target_type
        Target specifier, as passed to `train_network`.

    Returns
    -------
    int
        Best hyperparameter index.

    """

    _keep = lambda line: line.startswith('loss_ev')
    _extract = lambda line: float(line.strip().split(' = ')[1])

    errors, ks = [], []
    for fname in listdir(LOG_DIR):
        if not fname.startswith(f'train-{cls_name}-{target_type}'):
            continue

        with open(f'{LOG_DIR}/{fname}') as f:
            lines = filter(_keep, f.readlines())
            curve = list(map(_extract, lines))

        ks.append(int(fname.split('-')[-1].split('.')[0]))
        # errors.append(curve[-1])
        errors.append(min(curve))

    ks = torch.as_tensor(ks)
    errors = torch.as_tensor(errors)

    return ks[torch.argmin(errors)]

def _load_datasets(
    target_type: str,
    eval_data: str
) -> tuple[DataLoader, DataLoader]:
    """
    Load training, validation, and test sets, as defined by the indices saved in
    `DATA_DIR`. For convenience, each subset is returned in a `DataLoader`.

    Parameters
    ----------
    target_type
        Type of target data, as passed to `load_data`.
    eval_data
        Whether to hold out the `'validation'` dataset, in which case the first
        loader will contain only the training data, or the `'test'` dataset,
        such that the first loader contains training and validation data.

    Returns
    -------
    tuple[DataLoader, DataLoader]
        Triples of tensors (u, Y, Z) for training and held-out sets.

    """

    u, Y, targets = load_data(target_type)

    if target_type == 'Y-hat':
        from msgwam.dispersion import cp_x
        cp = cp_x(*Y.T[2:5])
        cp_hat = cp_x(*targets.T[2:5])

        targets = (cp_hat - cp)

        reject = abs(targets) > 5
        u, Y, targets = u[~reject], Y[~reject], targets[~reject, None]

    if target_type in ['fine', 'coarse']:
        idx_tr = torch.load(f'{DATA_DIR}/idx-tr.pkl')
        idx_va = torch.load(f'{DATA_DIR}/idx-va.pkl')
        idx_te = torch.load(f'{DATA_DIR}/idx-te.pkl')

    elif target_type == 'Y-hat':
        torch.manual_seed(7278)

        n_samples = u.shape[0]
        idx = torch.randperm(n_samples)
        a, b = int(0.7 * n_samples), int(0.85 * n_samples)
        idx_tr, idx_va, idx_te = idx[:a], idx[a:b], idx[b:]

    if eval_data == 'validation':
        idx_ev = idx_va

    elif eval_data == 'test':
        idx_tr = torch.cat((idx_tr, idx_va))
        idx_ev = idx_te
    
    noise = torch.normal(0, params.noise_level, size=u.shape)
    u[idx_tr] = u[idx_tr] + noise[idx_tr]

    
    # idx_tr = idx_tr[:10]

    print('Loaded ' + (
        f'{len(idx_tr)} training samples and ' +
        f'{len(idx_ev)} {eval_data} samples.'
    ))

    loaders = []
    for idx in (idx_tr, idx_ev):
        data = TensorDataset(u[idx], Y[idx], targets[idx])
        loaders.append(DataLoader(data, params.batch_size, shuffle=True))

    return tuple(loaders)

def _load_model(cls_name: str, restart: int) -> tuple[SourceNet, Adam]:
    """
    Load a model of the specified kind and an associated optimizer, and load any
    saved state for both modules if this is a restart run.

    Parameters
    ----------
    cls_name
        What kind of neural network to load. Must correspond to the (lowercase)
        name of a `SourceNet` subclass defined in `architectures`.
    restart
        What training restart this is. Zero corresponds to a new model.

    Returns
    -------
    SourceNet
        Requested subclass instance, with loaded state if necessary.
    Optimizer
        Associated optimizer, with loaded state if necessary.

    """

    model: SourceNet = getattr(architectures, cls_name.capitalize())()
    optimizer = Adam(model.parameters(), params.learning_rate)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model {params.task_id} has {n_params} trainable parameters.')

    if restart > 0:
        fname = f'state-{params.task_id}-r{restart - 1}.pkl'
        state = torch.load(f'{DATA_DIR}/{cls_name}-Y-hat/{fname}')

        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        print(f'Loaded state from training run {restart - 1}')

    return model, optimizer

def _trace_model(
    u: torch.Tensor,
    Y: torch.Tensor,
    model: SourceNet
) -> torch.jit.ScriptModule:
    """
    Trace a model so that it can be saved as a JITted function.

    Parameters
    ----------
    u
        Two-dimensional tensor of example zonal wind profiles.
    Y
        Two-dimensional tensor of example coarse ray volume properties.
    model
        Trained model to trace.

    Returns
    -------
    ScriptModule
        JITted model that can be saved to disc and called without access to the
        Python implementation of the class.

    """

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    with torch.no_grad():
        with catch_warnings(action='ignore', category=torch.jit.TracerWarning):
            if isinstance(model, Surrogate):
                traced = torch.jit.trace(model, (u, Y))

            elif isinstance(model, Tinkerer):
                def _to_trace(u_ex, Y_ex):
                    output = model(u_ex, Y_ex)
                    output = model._online_postprocess(u_ex, Y_ex, output)

                    return output

                traced = torch.jit.trace(_to_trace, (u, Y))

    return traced

def _train_epoch(
    loader: DataLoader,
    model: SourceNet,
    optimizer: Adam,
    loss_func: nn.Module
) -> float:
    """
    Run one epoch of model training.

    Parameters
    ----------
    loader
        Loader containing training inputs and targets.
    model
        `SourceNet` subclass instance to be trained.
    optimizer
        Optimizer updating the parameters of `model`.
    loss_func
        Module to compute loss between outputs and targets.

    Returns
    -------
    float
        Loss averaged over all samples in the training set.

    """

    model.train()
    weight_sum = 0
    total = 0

    for u, Y, targets in loader:
        optimizer.zero_grad()
        output = model(u, Y)
        weight = u.shape[0]

        loss = loss_func(targets, output)
        total = total + weight * loss.item()
        weight_sum = weight_sum + weight

        loss.backward()
        optimizer.step()

    return total / weight_sum

def _evaluate_epoch(
    loader: DataLoader,
    model: SourceNet,
    loss_func: nn.Module
) -> float:
    """
    Evaluate the model on the validation or test set.

    Parameters
    ----------
    model
        `SourceNet` subclass instance to evaluate.
    loader
        Loader containing validation or test inputs and targets.
    loss_func
        Module to compute loss between outputs and targets.

    Parameters
    ----------
    float
        Loss averaged over all samples in the validation or test set.

    """

    model.eval()
    with torch.no_grad():
        u, Y, targets = loader.dataset.tensors
        loss = loss_func(targets, model(u, Y))

    return loss.item()