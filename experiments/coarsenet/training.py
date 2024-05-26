from time import time

import torch

from torch.utils.data import DataLoader, TensorDataset

import hyperparameters as hparams

from architectures import CoarseNet
from utils import integrate_batches

MAX_HOURS = 5
N_BATCHES = 50
N_EPOCHS = 5

def train_network() -> None:
    """Train a CoarseNet instance."""

    loader_tr, loader_te = _load_data()
    *_, Z_tr = loader_tr.dataset.tensors
    stds = Z_tr.std(dim=(0, 1))

    model = CoarseNet()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=hparams.learning_rate,
        weight_decay=hparams.weight_decay
    )

    start = time()
    for n_epoch in range(N_EPOCHS):
        print(f'==== starting epoch {n_epoch + 1} ====')

        for i, (u, X, Z) in enumerate(loader_tr):
            batch_start = time()
            optimizer.zero_grad()
            output = model(u, X)

            spectrum = CoarseNet.build_spectrum(X, output)
            L2 = _l2_loss(u, spectrum, Z, stds)
            reg = _reg_loss(output)

            (hparams.beta * reg + L2).backward()
            optimizer.step()

            batch_time = time() - batch_start
            message = f'batch {i + 1} ({batch_time:.2f} seconds): '
            message += f'L2 = {L2.item():.4f} reg = {reg.item():.4f}'
            print(message)

            del L2, reg
            if i + 1 == N_BATCHES:
                break

        with torch.no_grad():
            model.eval()
            total_L2 = 0
            total_reg = 0

            for i, (u, X, Z) in enumerate(loader_te):
                output = model(u, X)
                spectrum = CoarseNet.build_spectrum(X, output)
                L2 = _l2_loss(u, spectrum, Z, stds)
                reg = _reg_loss(output)

                total_L2 = total_L2 + L2.item()
                total_reg = total_reg + reg.item()

            avg_L2 = total_L2 / len(loader_te)
            avg_reg = total_reg / len(loader_te)

            print(f'\ntest L2 error  = {avg_L2:.4f}')
            print(f'test reg error = {avg_reg:.4f}\n')
            model.train()

        hours = (time() - start) / 3600
        if hours > MAX_HOURS:
            break

    path = f'data/coarsenet/model-{hparams.task_id}.pkl'
    torch.save(model.state_dict(), path)

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
    Z = torch.load('data/coarsenet/targets.pkl').mean(dim=2)
    u = torch.load('data/coarsenet/wind.pkl')[:, 0]

    m = int(0.8 * X.shape[2])
    idx = torch.randperm(X.shape[2])
    idx_tr, idx_te = idx[:m], idx[m:]

    print('==== separating training and test data ====')
    print('idx_tr:', idx_tr.tolist())
    print('idx_te:', idx_te.tolist(), '\n')

    data_tr = TensorDataset(u, X[:, :, idx_tr], Z[:, idx_tr])
    data_te = TensorDataset(u, X[:, :, idx_te], Z[:, idx_te])
    loader_tr = DataLoader(data_tr, batch_size=None, shuffle=True)
    loader_te = DataLoader(data_te, batch_size=None, shuffle=True)

    return loader_tr, loader_te

def _l2_loss(
    u: torch.Tensor,
    spectrum: torch.Tensor,
    Z: torch.Tensor,
    stds: torch.Tensor
) -> torch.Tensor:
    """
    Compute the mean-square error in integration output.

    Parameters
    ----------
    u
        Fixed zonal wind profile to use during integration.
    spectrum
        Spectrum of wave packets to launch, as returned by the `build_spectrum`
        method of `CoarseNet`.
    Z
        Target momentum fluxes at each level, averaged over the integration.
    stds
        Standard deviation of the momentum flux at each level, used to scale the
        loss to interpretable units.

    Returns
    -------
    torch.Tensor
        Mean-square error over all wave packets and levels.

    """

    wind = torch.vstack((u, torch.zeros_like(u)))
    Z_hat = integrate_batches(
        wind, spectrum,
        rays_per_packet=1,
        smooth=True
    ).mean(dim=1)

    idx = stds != 0
    errors = torch.zeros_like(Z)
    errors[:, idx] = (Z_hat - Z)[:, idx] / stds[idx]

    return (errors ** 2).mean()

def _reg_loss(output: torch.Tensor) -> torch.Tensor:
    """
    Regularization loss based on some notion of physical plausibility of the
    replacement ray volume. Currently penalizes deviation of the replacement ray
    from the one obtained by simply taking the mean of each ray property.

    Parameters
    ----------
    output
        Neural network output, before rescaling by physical dimensions.

    Returns
    -------
    torch.Tensor
        Regularization loss.

    """

    return ((output - 1) ** 2).mean()