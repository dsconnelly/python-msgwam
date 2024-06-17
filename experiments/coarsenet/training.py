from time import time
from warnings import catch_warnings

import torch

from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils import clip_grad_norm_

from architectures import CoarseNet
from hyperparameters import (
    beta_decay,
    learning_rate,
    smoothing_decay,
    task_id,
    weight_decay
)
from losses import RegularizedMSELoss

DATA_DIR = 'data/coarsenet'
RESTART = 0

MAX_BATCHES = 50
MAX_EPOCHS = 75
MAX_HOURS = 11

MAX_GRAD_NORM = 2
MAX_SMOOTHING = 16
MIN_SMOOTHING = 2

def train_network() -> None:
    """Train a CoarseNet instance."""

    loss = RegularizedMSELoss()
    loader_tr, loader_va = _load_data(loss.keep)
    u_tr, Y_tr, _ = loader_tr.dataset.tensors

    model = CoarseNet(u_tr, Y_tr)
    optimizer = _get_optimizer(model)

    if RESTART > 0:
        state = torch.load(f'{DATA_DIR}/state-{task_id}.pkl')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])

        print(f'Resuming training of model {task_id}\n')

    hours = 0
    start = time()
    n_epoch = 1

    while n_epoch <= MAX_EPOCHS and hours < MAX_HOURS:
        print(f'==== starting epoch {n_epoch} ====')

        arg = torch.tensor(RESTART + n_epoch - 1)
        smoothing = MAX_SMOOTHING * torch.exp(-arg / smoothing_decay)
        beta = torch.exp(-arg / beta_decay)

        tail = f'{smoothing:.4f}'
        if smoothing < MIN_SMOOTHING:
            smoothing = None
            tail = 'None'

        print(f'smoothing = {tail}')
        print(f'beta = {beta:.4f}\n')

        _train(model, loader_tr, optimizer, loss, smoothing, beta)
        _ = _validate(model, loader_va, loss)

        n_epoch = n_epoch + 1
        hours = (time() - start) / 3600

    for param in model.parameters():
        param.detach_()

    def _evaluate(u: torch.Tensor, X: torch.Tensor):
        return model.build_spectrum(X, model(u, X))
    
    with catch_warnings(action='ignore', category=torch.jit.TracerWarning):
        with torch.no_grad():
            *example_inputs, _ = next(iter(loader_tr))
            traced = torch.jit.trace(_evaluate, example_inputs)

    state = {
        'model' : model.state_dict(),
        'optimizer' : optimizer.state_dict()
    }

    torch.jit.save(traced, f'{DATA_DIR}/model-{task_id}.jit')
    torch.save(state, f'{DATA_DIR}/state-{task_id}.pkl')

def _get_optimizer(model: CoarseNet) -> torch.optim.Optimizer:
    """
    Get a new optimizer for the model being trained.

    Parameters
    ----------
    model
        Model being trained

    Returns
    -------
    torch.optim.Optimizer
        New optimizer with correct hyperparameter settings.

    """

    return torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=(weight_decay * learning_rate)
    )

def _load_data(keep: torch.Tensor) -> tuple[DataLoader, DataLoader]:
    """
    Construct `DataLoader` objects containing training and validation datasets,
    keeping only the requested vertical levels.

    Parameters
    ----------
    keep
        Boolean index indicating which levels should be included in the targets.

    Returns
    -------
    DataLoader
        Training (u, X, Z) triples.
    DataLoader
        Validation (u, X, Z) triples.

    """

    u = torch.load(f'{DATA_DIR}/wind.pkl')[:, 0, 0]
    Y = torch.load(f'{DATA_DIR}/squares.pkl')
    Z = torch.load(f'{DATA_DIR}/targets.pkl')
    Z = Z.mean(dim=2)[..., keep]

    if RESTART > 0:
        idx_tr = torch.load(f'{DATA_DIR}/idx-tr-{task_id}.pkl')
        idx_va = torch.load(f'{DATA_DIR}/idx-va-{task_id}.pkl')

    else:
        m = int(0.8 * Y.shape[2])
        idx = torch.randperm(Y.shape[2])
        idx_tr, idx_va = idx[:m], idx[m:]

        torch.save(idx_tr, f'{DATA_DIR}/idx-tr-{task_id}.pkl')
        torch.save(idx_va, f'{DATA_DIR}/idx-va-{task_id}.pkl')

    data_tr = TensorDataset(u, Y[:, :, idx_tr], Z[:, idx_tr])
    data_va = TensorDataset(u, Y[:, :, idx_va], Z[:, idx_va])
    loader_tr = DataLoader(data_tr, batch_size=None, shuffle=True)
    loader_va = DataLoader(data_va, batch_size=None, shuffle=True)

    return loader_tr, loader_va

def _train(
    model: CoarseNet,
    loader: DataLoader,
    optimizer: torch.optim.Adam,
    loss: RegularizedMSELoss,
    smoothing: float,
    beta: float,
) -> None:
    """
    Evaluate the training losses and backpropagate.

    Parameters
    ----------
    model
        `CoarseNet instance being trained.
    loader
        `DataLoader` containing training data.
    optimizer
        Optimizer handling gradient updates.
    loss
        Module calculating regularization and MSE losses.
    smoothing
        Current smoothing value to use during integration.
    n_batches
        How many batches to train for this epoch. 

    """
    
    model.train()
    for i, (u, Y, Z) in enumerate(loader):
        start = time()
        optimizer.zero_grad()
        reg, mse = loss(u, Y, Z, model, smoothing)
        total = beta * reg + (1 - beta) * mse

        total.backward()
        clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()

        runtime = time() - start
        message = f'batch {i + 1} ({runtime:.2f} seconds): '
        message += f'reg = {reg.item():.4f} mse = {mse.item():.4f}'
        print(message)

        del reg, mse, total
        if i + 1 == MAX_BATCHES:
            return

def _validate(
    model: CoarseNet,
    loader: DataLoader,
    loss: RegularizedMSELoss,
) -> float:
    """
    Evaluate the model on the validation data and return the MSE loss.

    Parameters
    ----------
    model
        `CoarseNet instance being trained.
    loader
        `DataLoader` containing training data.
    loss
        Module calculating regularization and MSE losses.

    Returns
    -------
    float
        MSE loss averaged over all batches in the validation set.

    """

    model.eval()
    with torch.no_grad():
        reg_va = 0
        mse_va = 0

        for k, (u, Y, Z) in enumerate(loader):
            reg, mse = loss(u, Y, Z, model, None)
            reg_va = reg_va + reg.item()
            mse_va = mse_va + mse.item()

            if k + 1 == MAX_BATCHES:
                break

    reg_va = reg_va / (k + 1)
    mse_va = mse_va / (k + 1)

    print(f'\nvalidation reg error = {reg_va:.4f}')
    print(f'validation MSE error = {mse_va:.4f}\n')

    return mse_va