from time import time

import torch

from architectures import CoarseNet
from utils import integrate_batches

BETA = 0
MAX_HOURS = 22
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
            L2, reg = _loss(u_batch, X_batch, output, Z_batch)
            (reg + L2).backward()
            optimizer.step()

            batch_time = time() - batch_start
            message = f'    batch {i + 1} ({batch_time:.2f} seconds): '
            message = message + f'L2 error = {L2.item():.4g}'
            print(message)

        with torch.no_grad():
            model.eval()
            total_L2 = 0
            total_reg = 0

            for u_batch, X_batch, Z_batch in zip(u_te, X_te, Z_te):
                output = model(u_batch, X_batch)
                L2, reg = _loss(u_batch, X_batch, output, Z_batch)

                total_L2 = total_L2 + L2.item()
                total_reg = total_reg + reg.item()

            print(' ' * 8 + f'test L2 error  = {total_L2:.4g}')
            print(' ' * 8 + f'test reg error = {total_reg:.4g}')
            model.train()

        hours = (time() - start) / 3600
        if hours > MAX_HOURS:
            break

    torch.save(model.state_dict(), 'data/coarsenet/model.pkl')

def _loss(
    u: torch.Tensor,
    X: torch.Tensor,
    output: torch.Tensor,
    Z: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate the squared error between the pseudomomentum fluxes associated
    with the original wave packet and those from the replacement ray volume,
    summed over the duration of the integration.

    Parameters
    ----------
    u
        Fixed zonal wind profile to use during integration.
    X
        Original wave packet properties.
    output
        Replacement ray volume properties as returned by a `CoarseNet`.
    Z
        Correct online pseudomomentum fluxes for each wave packet.

    Returns
    -------
    torch.Tensor
        Squared error in momentum flux for each wave packet.
    torch.Tensor
        Regularization error penalizing deviations from the mean packet size.

    """
    
    wind = torch.vstack((u, torch.zeros_like(u)))
    spectrum = CoarseNet.build_spectrum(X, output)
    Z_hat = integrate_batches(wind, spectrum, 1).mean(dim=1)
    L2 = ((Z - Z_hat) ** 2).sum()

    means = torch.nanmean(X[CoarseNet.idx], dim=2)
    reg = BETA * (((output / means) - 1) ** 2).sum()

    return L2, reg
