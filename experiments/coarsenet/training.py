from time import time
from warnings import catch_warnings

import torch

from torch.utils.data import DataLoader, TensorDataset

from architectures import CoarseNet
from hyperparameters import beta, learning_rate, task_id, weight_decay
from losses import RegularizedMSELoss
from monitors import LossMonitor

MAX_BATCHES = 50
MAX_EPOCHS = 100
MAX_HOURS = 9

MAX_SMOOTHING = 16
MIN_SMOOTHING = 4

def train_network() -> None:
    """Train a CoarseNet instance."""

    model = CoarseNet()
    loss = RegularizedMSELoss()
    monitor = LossMonitor(6, 0.04)
    loader_tr, loader_va = _load_data(loss.keep)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    hours = 0
    n_epoch = 1
    smoothing = MAX_SMOOTHING

    start = time()
    while n_epoch <= MAX_EPOCHS and hours < MAX_HOURS:
        print(f'==== starting epoch {n_epoch} ====')

        _train(model, loader_tr, optimizer, loss, smoothing)
        mse_va = _validate(model, loader_va, loss, smoothing)
        
        if monitor.has_plateaued(mse_va):
            if smoothing <= MIN_SMOOTHING:
                break

            smoothing = smoothing // 2
            print(f'Reducing smoothing to {smoothing}')

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

    torch.jit.save(traced, f'data/coarsenet/model-{task_id}.jit')
    torch.save(model.state_dict(), f'data/coarsenet/model-{task_id}.pkl')

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

    u = torch.load('data/coarsenet/wind.pkl')[:, 0]
    X = torch.load('data/coarsenet/packets.pkl')
    Z = torch.load('data/coarsenet/targets.pkl')
    Z = Z.mean(dim=2)[..., keep]

    m = int(0.8 * X.shape[2])
    idx = torch.randperm(X.shape[2])
    idx_tr, idx_va = idx[:m], idx[m:]

    print('==== separating training and validation data')
    print(f'idx_tr: {idx_tr.tolist()}')
    print(f'idx_va: {idx_va.tolist()}\n')

    data_tr = TensorDataset(u, X[:, :, idx_tr], Z[:, idx_tr])
    data_va = TensorDataset(u, X[:, :, idx_va], Z[:, idx_va])
    loader_tr = DataLoader(data_tr, batch_size=None, shuffle=True)
    loader_va = DataLoader(data_va, batch_size=None, shuffle=True)

    return loader_tr, loader_va

def _train(
    model: CoarseNet,
    loader: DataLoader,
    optimizer: torch.optim.Adam,
    loss: RegularizedMSELoss,
    smoothing: float
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

    """
    
    model.train()
    for i, (u, X, Z) in enumerate(loader):
        start = time()
        optimizer.zero_grad()

        reg, mse = loss(u, X, Z, model, smoothing)
        (beta * reg + mse).backward()
        optimizer.step()

        runtime = time() - start
        message = f'batch {i + 1} ({runtime:.2f} seconds): '
        message += f'reg = {reg.item():.4f} mse = {mse.item():.4f}'
        print(message)

        del reg, mse
        if i + 1 == MAX_BATCHES:
            return

def _validate(
    model: CoarseNet,
    loader: DataLoader,
    loss: RegularizedMSELoss,
    smoothing: float
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
    smoothing
        Current smoothing value to use during integration.

    Returns
    -------
    float
        MSE loss averaged over all batches in the validation set.

    """

    model.eval()
    with torch.no_grad():
        reg_va = 0
        mse_va = 0

        for u, X, Z in loader:
            reg, mse = loss(u, X, Z, model, smoothing)
            reg_va = reg_va + reg.item()
            mse_va = mse_va + mse.item()

    reg_va = reg_va / len(loader)
    mse_va = mse_va / len(loader)

    print(f'\nvalidation reg error = {reg_va:.4f}')
    print(f'validation MSE error = {mse_va:.4f}\n')

    return mse_va