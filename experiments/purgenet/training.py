from time import time

import torch

from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from architectures import PurgeNet
from hyperparameters import batch_size, learning_rate, weight_decay, task_id
from utils import standardize

DATA_DIR = 'data/purgenet'
RESUME = False

MAX_EPOCHS = 20
MAX_HOURS = 5

def train_network() -> None:
    """Train a `PurgeNet` instance."""

    loader_tr, loader_va = _load_data()
    model = PurgeNet(loader_tr.dataset.tensors[0])
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=learning_rate,
        weight_decay=(weight_decay * learning_rate)
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model has {n_params} trainable parameters\n')

    if RESUME:
        print(f'Resuming training model {task_id}\n')

        state = torch.load(f'{DATA_DIR}/state-{task_id}.pkl')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])

    hours = 0
    start = time()
    n_epoch = 1

    while n_epoch <= MAX_EPOCHS and hours < MAX_HOURS:
        model.train()

        total_tr = 0
        for Y, Z in loader_tr:
            optimizer.zero_grad()
            loss = _loss(model, Y, Z)
            total_tr += loss.item() * Y.shape[0]

            loss.backward()
            optimizer.step()

        with torch.no_grad():
            model.eval()
            
            total_va = 0
            for Y, Z in loader_va:
                loss = _loss(model, Y, Z)
                total_va += loss.item() * Y.shape[0]

            total_tr /= len(loader_tr.dataset)
            total_va /= len(loader_va.dataset)

            print(f'==== epoch {n_epoch} ====')
            print(f'loss_tr = {total_tr:.3f}')
            print(f'loss_va = {total_va:.3f}\n')

        n_epoch = n_epoch + 1
        hours = (time() - start) / 3600

    for param in model.parameters():
        param.detach_()

    with torch.no_grad():
        model.eval()
        Y_ex = loader_tr.dataset.tensors[0][:256]
        traced = torch.jit.trace(model, Y_ex)

    state = {
        'model' : model.state_dict(),
        'optimizer' : optimizer.state_dict()
    }

    torch.jit.save(traced, f'{DATA_DIR}/model-{task_id}.jit')
    torch.save(state, f'{DATA_DIR}/state-{task_id}.pkl')

def _loss(model: PurgeNet, Y: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
    """
    Calculate the MSE loss for a given neural network and input-target pair.

    Parameters
    ----------
    model
        Neural network being trained.
    Y
        Tensor of inputs.
    Z
        Tensor of targets.

    Returns
    -------
    torch.Tensor
        Mean squared error over all inputs and target channels.

    """

    return ((model(Y) - Z) ** 2).mean()

def _load_data() -> tuple[DataLoader, DataLoader]:
    """
    Load the data, split it into training and validation sets, and standardize
    the outputs by vertical level.

    Returns
    -------
    DataLoader
        Training (Y, Z) pairs.
    DataLoader
        Validation (Y, Z) pairs.

    """
    
    Y = torch.load(f'{DATA_DIR}/Y.pkl')
    Z = torch.load(f'{DATA_DIR}/Z.pkl')

    drop = torch.isnan(Y).sum(dim=1) > 1
    Y, Z = Y[~drop], Z[~drop]

    m = int(0.8 * Y.shape[0])
    idx = torch.randperm(Y.shape[0])
    idx_tr, idx_va = idx[:m], idx[m:]

    Y_tr, Y_va = Y[idx_tr], Y[idx_va]
    Z_tr, Z_va = Z[idx_tr], Z[idx_va]

    n_tr, n_va = len(idx_tr), len(idx_va)
    print(f'Loaded {n_tr} training samples and {n_va} validation samples\n')

    means, stds = Z_tr.mean(dim=0), Z_tr.std(dim=0)
    Z_tr = standardize(Z_tr, means, stds)
    Z_va = standardize(Z_va, means, stds)

    dataset_tr = TensorDataset(Y_tr, Z_tr)
    dataset_te = TensorDataset(Y_va, Z_va)
    loader_tr = DataLoader(dataset_tr, batch_size, shuffle=True)
    loader_te = DataLoader(dataset_te, 4096, shuffle=False)

    return loader_tr, loader_te
