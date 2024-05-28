from time import time

import torch

from torch.utils.data import DataLoader, TensorDataset

from architectures import CoarseNet
from hyperparameters import beta, learning_rate, task_id, weight_decay
from utils import integrate_batches

MAX_HOURS = 5
N_BATCHES = 5
N_EPOCHS = 50

_faces = torch.linspace(0, 60, 101)
_centers = (_faces[:-1] + _faces[1:]) / 2
KEEP = (15 < _centers) & (_centers < 50)

def train_network() -> None:
    """Train a CoarseNet instance."""

    model = CoarseNet()
    scheduler = Scheduler(6, 0.04)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    state = torch.load(f'data/coarsenet/model-{task_id}.pkl')
    model.load_state_dict(state)

    loader_tr, loader_va = _load_data()

    hours = 0
    n_epoch = 1
    smoothing = 16

    start = time()
    while n_epoch <= N_EPOCHS and hours < MAX_HOURS:
        print(f'==== starting epoch {n_epoch} ====')
        _train(model, optimizer, loader_tr, smoothing)
        L2_va = _validate(model, loader_va, smoothing)

        if scheduler.is_stuck(L2_va):
            if smoothing == 4:
                break

            smoothing = smoothing // 2
            print(f'Reducing smoothing to {smoothing}\n')
        
        n_epoch = n_epoch + 1
        hours = (time() - start) / 3600
        
    path = f'data/coarsenet/model-{task_id}.pkl'
    torch.save(model.state_dict(), path)

class Scheduler:
    def __init__(self, patience: int, min_decrease: float) -> None:
        """
        Initialize a `Scheduler` tracking the validation loss.

        Parameters
        ----------
        patience
            How many epochs the the validation loss can not decrease before the
            scheduler considers itself stuck.
        min_decrease
            Percentage by which the validation loss must decrease for the epoch
            not to increase the waiting count.

        """

        self.patience = patience
        self.min_decrease = min_decrease
        self._reset(torch.inf)

    def is_stuck(self, L2_va: float) -> bool:
        """
        Determine whether the validation loss has plateaued.

        Parameters
        ----------
        L2_va
            Validation loss from the current epoch.

        Returns
        -------
        bool
            Whether the validation loss has plateaued.

        """

        if L2_va > 1 or L2_va <= (1 - self.min_decrease) * self.last_L2_va:
            self._reset(L2_va)
            return False
        
        self.n_waiting = self.n_waiting + 1
        if self.n_waiting == self.patience:
            self._reset(L2_va)
            return True
        
        return False
        
    def _reset(self, L2_va: float) -> None:
        """
        Reset the scheduler state to start from the current validation loss.

        Parameters
        ----------
        L2_va
            Validation loss from the current epoch.

        """
        
        self.last_L2_va = L2_va
        self.n_waiting = 0

def _load_data() -> tuple[DataLoader, DataLoader]:
    """
    Construct `DataLoader` objects containing training and test datasets.

    Returns
    -------
    DataLoader
        Training (u, X, Z) triples.
    DataLoader
        Test (u, X, Z) triples.

    """

    X = torch.load('data/coarsenet/packets.pkl')
    Z = torch.load('data/coarsenet/targets.pkl')
    u = torch.load('data/coarsenet/wind.pkl')[:, 0]
    Z = Z.mean(dim=2)[..., KEEP]

    m = int(0.8 * X.shape[2])
    idx = torch.randperm(X.shape[2])
    idx_tr, idx_te = idx[:m], idx[m:]

    print('==== separating training and test data ====')
    print('idx_tr:', idx_tr.tolist())
    print('idx_te:', idx_te.tolist(), '\n')

    data_tr = TensorDataset(u, X[:, :, idx_tr], Z[:, idx_tr])
    data_te = TensorDataset(u, X[:, :, idx_te], Z[:, idx_te])
    loader_tr = DataLoader(data_tr, batch_size=None, shuffle=False)
    loader_te = DataLoader(data_te, batch_size=None, shuffle=False)

    return loader_tr, loader_te

def _get_losses(
    model: CoarseNet,
    u: torch.Tensor,
    X: torch.Tensor,
    Z: torch.Tensor,
    smoothing: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate both the regularization and L2 losses.

    Parameters
    ----------
    model
        The `CoarseNet` instance being trained.
    u
        Zonal wind profile for the given batch.
    X
        Wave packets for the given batch.
    Z
        Mean momentum flux profiles for the given batch.
    smoothing
        Smoothing value to use in Gaussian projection.
    
    Returns
    -------
    torch.Tensor
        Regularization loss. Currently penalizes squared deviations from the
        default replacement ray volumes.
    torch.Tensor
        L2 loss between integrated neural network outputs and target fluxes.

    """

    output = model(u, X)
    reg = ((output - 1) ** 2).mean()

    spectrum = model.build_spectrum(X, output)
    wind = torch.vstack((u, torch.zeros_like(u)))

    Z_hat = integrate_batches(
        wind, spectrum,
        rays_per_packet=1,
        smoothing=smoothing
    ).mean(dim=1)[..., KEEP]

    scales, _ = abs(Z).max(dim=-1)
    errors = (Z_hat - Z) / scales[:, None]
    L2 = (errors ** 2).mean()

    return reg, L2

def _train(
    model: CoarseNet,
    optimizer: torch.optim.Adam,
    loader: DataLoader,
    smoothing: float
) -> None:
    """
    Evaluate the training losses and backpropagate.

    Parameters
    ----------
    model
        `CoarseNet` instance being trained.
    optimizer
        Optimizer handling gradient updates.
    loader
        `DataLoader` containing training data.
    smoothing
        Current smoothing value to use during integration.

    """
    
    model.train()

    for i, (u, X, Z) in enumerate(loader):    
        batch_start = time()
        optimizer.zero_grad()

        reg, L2 = _get_losses(model, u, X, Z, smoothing)
        (beta * reg + L2).backward()
        optimizer.step()

        batch_time = time() - batch_start
        message = f'batch {i + 1} ({batch_time:.2f} seconds): '
        message += f'reg = {reg.item():.4f} L2 = {L2.item():.4f}'
        print(message)

        del reg, L2
        if i + 1 == N_BATCHES:
            return

def _validate(
    model: CoarseNet,
    loader: DataLoader,
    smoothing: float
) -> float:
    """
    Evaluate the model on the validation data and return the loss.

    Parameters
    ----------
    model
        `CoarseNet` instance being trained.
    loader
        `DataLoader` containing validation data.
    smoothing
        Current smoothing value to use during integration.

    Returns
    -------
    float
        L2 loss averaged over all batches in the validation set.
    
    """
    
    model.eval()
    with torch.no_grad():
        total_reg, total_L2 = 0, 0

        c = 0
        for u, X, Z in loader:
            reg, L2 = _get_losses(model, u, X, Z, smoothing)
            total_reg = total_reg + reg.item()
            total_L2 = total_L2 + L2.item()

            c += 1
            if c == N_BATCHES:
                break

    reg_va = total_reg / N_BATCHES# len(loader)
    L2_va = total_L2 / N_BATCHES# len(loader)

    print(f'\nvalidation reg error  = {reg_va:.4f}')
    print(f'validation L2 error = {L2_va:.4f}\n')

    return L2_va