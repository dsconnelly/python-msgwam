from time import time

import torch

from architectures import CoarseNet
from utils import integrate_batches

BETA = 0.3
MAX_HOURS = 4
N_EPOCHS = 45

def train_network() -> None:
    """Train a CoarseNet instance."""

    wind = torch.load('data/coarsenet/wind.pkl')
    X = torch.load('data/coarsenet/packets.pkl')
    Z = torch.load('data/coarsenet/targets.pkl').mean(dim=2)
    u = wind[:, 0]

    m = int(0.8 * X.shape[2])
    idx = torch.randperm(X.shape[2])
    idx_tr, idx_te = idx[:m], idx[m:]

    u_tr, u_te = u, u
    X_tr, X_te = X[:, :, idx_tr], X[:, :, idx_te]
    Z_tr, Z_te = Z[:, idx_tr], Z[:, idx_te]
    stds = Z_tr.std(dim=(0, 1))

    model = CoarseNet()
    params = model.parameters()
    optimizer = torch.optim.Adam(params, lr=0.001)

    start = time()
    for n_epoch in range(N_EPOCHS):
        print(f'==== starting epoch {n_epoch + 1} ====')
        for i, (u_batch, X_batch, Z_batch) in enumerate(zip(u_tr, X_tr, Z_tr)):
            batch_start = time()
            optimizer.zero_grad()

            output = model(u_batch, X_batch)
            spectrum = CoarseNet.build_spectrum(X_batch, output)
            L2 = _l2_loss(u_batch, spectrum, Z_batch, stds)
            reg = _reg_loss(output)
            (reg + L2).backward()
            optimizer.step()

            batch_time = time() - batch_start
            message = f'    batch {i + 1} ({batch_time:.2f} seconds): '
            message += f'L2 = {L2.item():.4f} reg = {reg.item():.4f}'
            print(message)

        with torch.no_grad():
            model.eval()
            total_L2 = 0
            total_reg = 0

            for u_batch, X_batch, Z_batch in zip(u_te, X_te, Z_te):
                output = model(u_batch, X_batch)
                spectrum = CoarseNet.build_spectrum(X_batch, output)
                L2 = _l2_loss(u_batch, spectrum, Z_batch, stds)
                reg = _reg_loss(output)

                total_L2 = total_L2 + L2.item()
                total_reg = total_reg + reg.item()

            print(' ' * 8 + f'test L2 error  = {total_L2:.4f}')
            print(' ' * 8 + f'test reg error = {total_reg:.4f}\n')
            model.train()

        hours = (time() - start) / 3600
        if hours > MAX_HOURS:
            break

    torch.save(model.state_dict(), 'data/coarsenet/model.pkl')

def _l2_loss(
    u: torch.Tensor,
    spectrum: torch.Tensor,
    Z: torch.Tensor,
    stds: torch.Tensor
) -> torch.Tensor:
    """
    _summary_

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
    Z_hat = integrate_batches(wind, spectrum, 1).mean(dim=1)

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

    return BETA * ((output - 1) ** 2).mean()