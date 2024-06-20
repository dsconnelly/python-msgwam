from time import time
from warnings import catch_warnings

import torch

from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils import clip_grad_norm_

from architectures import CoarseNet
from hyperparameters import (
    learning_rate,
    task_id,
    use_adam,
    weight_decay
)
from losses import RegularizedMSELoss

DATA_DIR = 'data/coarsenet'
RESTART = False

MAX_BATCHES = 45
MAX_EPOCHS = 20
MAX_HOURS = 11

MAX_GRAD_NORM = 5

def train_network() -> None:
    """Train a CoarseNet instance."""

    loader_tr, loader_va = _load_data()
    u_tr, Y_tr, _ = loader_tr.dataset.tensors

    model = CoarseNet(u_tr[:, 0], Y_tr)
    optimizer = _get_optimizer(model)
    loss = RegularizedMSELoss()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model has {n_params} trainable parameters')

    if RESTART:
        state = torch.load(f'{DATA_DIR}/state-{task_id}.pkl')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])

        print(f'Resuming training of model {task_id}\n')

    else:
        n_reg = _regularize(model, loader_tr, optimizer, 5e-6)
        optimizer = _get_optimizer(model)

        print(f'Regularized model {task_id} in {n_reg} epochs\n')

    hours = 0
    start = time()
    n_epoch = 1

    while n_epoch <= MAX_EPOCHS and hours < MAX_HOURS:
        print(f'==== starting epoch {n_epoch} ====\n')
        _train(model, loader_tr, optimizer, loss)
        _validate(model, loader_va, loss)

        n_epoch = n_epoch + 1
        hours = (time() - start) / 3600

    for param in model.parameters():
        param.detach_()

    def _evaluate(u: torch.Tensor, Y: torch.Tensor):
        return model.build_spectrum(Y, model(u, Y))
    
    with catch_warnings(action='ignore', category=torch.jit.TracerWarning):
        with torch.no_grad():
            u_ex, Y_ex, _ = next(iter(loader_tr))
            traced = torch.jit.trace(_evaluate, (u_ex[0], Y_ex))

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

    cls = [torch.optim.SGD, torch.optim.Adam][use_adam]
    decay = weight_decay * learning_rate

    return cls(model.parameters(), lr=learning_rate, weight_decay=decay)

def _load_data() -> tuple[DataLoader, DataLoader]:
    """
    Construct `DataLoader` objects containing training and validation datasets.

    Returns
    -------
    DataLoader
        Training (u, X, Z) triples.
    DataLoader
        Validation (u, X, Z) triples.

    """

    u = torch.load(f'{DATA_DIR}/wind.pkl')[:, :, 0]
    Y = torch.load(f'{DATA_DIR}/squares.pkl')
    Z = torch.load(f'{DATA_DIR}/targets.pkl')
    Z = Z.mean(dim=2)

    if RESTART:
        idx_tr = torch.load(f'{DATA_DIR}/idx-tr-{task_id}.pkl')
        idx_va = torch.load(f'{DATA_DIR}/idx-va-{task_id}.pkl')

    else:
        m = int(0.8 * Y.shape[0])
        idx = torch.randperm(Y.shape[0])
        idx_tr, idx_va = idx[:m], idx[m:]

        torch.save(idx_tr, f'{DATA_DIR}/idx-tr-{task_id}.pkl')
        torch.save(idx_va, f'{DATA_DIR}/idx-va-{task_id}.pkl')

    data_tr = TensorDataset(u[idx_tr], Y[idx_tr], Z[idx_tr])
    data_va = TensorDataset(u[idx_va], Y[idx_va], Z[idx_va])
    loader_tr = DataLoader(data_tr, batch_size=None, shuffle=True)
    loader_va = DataLoader(data_va, batch_size=None, shuffle=True)

    return loader_tr, loader_va

def _regularize(
    model: CoarseNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    threshold: float
) -> int:
    """
    Prepare a model for training by training it to predict the properties of the
    square coarse ray volumes.

    Parameters
    ----------
    model
        `CoarseNet` instance to be trained.
    loader
        Training (u, Y, Z) triples.
    optimizer
        Optimizer to use during regularization. Should probably be discarded
        after this function is called.
    threshold
        Dataset-averaged regularization error below which regularization will be
        considered complete.

    Returns
    -------
    int
        Number of training epochs necessary to achieve regularization.

    """

    total = 1
    n_epochs = 0

    model.train()
    while total > threshold:
        total = 0

        for u, Y, _ in loader:
            optimizer.zero_grad()
            output = model(u[0], Y)

            reg = ((output - 1) ** 2).mean()
            total = total + reg.item()
            reg.backward()
            optimizer.step()

        total = total / len(loader)
        n_epochs = n_epochs + 1

    return n_epochs

def _train(
    model: CoarseNet,
    loader: DataLoader,
    optimizer: torch.optim.Adam,
    loss: RegularizedMSELoss,
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

    """
    
    model.train()
    for i, (u, Y, Z) in enumerate(loader):
        start = time()
        optimizer.zero_grad()
        mse = loss(u, Y, Z, model)

        mse.backward()
        clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()

        runtime = time() - start
        message = f'batch {i + 1} ({runtime:.2f} seconds): '
        message += f'loss = {mse.item():.4f}'
        print(message)

        del mse
        if i + 1 == MAX_BATCHES:
            return

def _validate(
    model: CoarseNet,
    loader: DataLoader,
    loss: RegularizedMSELoss,
) -> None:
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

    """

    model.eval()
    with torch.no_grad():
        mse = 0

        for k, (u, Y, Z) in enumerate(loader):
            mse = mse + loss(u, Y, Z, model).item()

            if k + 1 == MAX_BATCHES:
                break

    mse = mse / (k + 1)
    print(f'\nvalidation error = {mse:.4f}\n')
