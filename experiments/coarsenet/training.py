from time import time

import torch

from architectures import CoarseNet
from utils import integrate_batches

MAX_HOURS = 22
N_EPOCHS = 45

def train_network() -> None:
    """Train a CoarseNet instance."""

    wind = torch.load('data/coarsenet/wind.pkl')
    X = torch.load('data/coarsenet/packets.pkl')
    Z = torch.load('data/coarsenet/targets.pkl')
    u = wind[:, 0]

    m = int(0.8 * X.shape[0])
    idx = torch.randperm(X.shape[0])
    idx_tr, idx_te = idx[:m], idx[m:]

    u_tr, u_te = u[idx_tr], u[idx_te]
    X_tr, X_te = X[idx_tr], X[idx_te]
    Z_tr, Z_te = Z[idx_tr], Z[idx_te]
    
    model = CoarseNet()
    params = model.parameters()
    optimizer = torch.optim.Adam(params, lr=0.005)

    start = time()
    for n_epoch in range(N_EPOCHS):
        for u_batch, X_batch, Z_batch in zip(u_tr, X_tr, Z_tr):
            optimizer.zero_grad()
            output = model(u_batch, X_batch)
            grad = _grad_loss(u_batch, X_batch, output, Z_batch)

            output.backward(grad)
            optimizer.step()

        with torch.no_grad():
            model.eval()

            loss = 0
            for u_batch, X_batch, Z_batch in zip(u_te, X_te, Z_te):
                output = model(u_batch, X_batch)
                loss = loss + _loss(u_batch, X_batch, output, Z).sum().item()

            print(f'epoch {n_epoch + 1}: loss = {loss:.4g}')
            model.train()

        hours = (time() - start) / 60
        if hours > MAX_HOURS:
            break

    torch.save(model.state_dict(), 'data/coarsenet/model.pkl')

def _grad_loss(
    u: torch.Tensor,
    X: torch.Tensor,
    output: torch.Tensor,
    Z: torch.Tensor
) -> torch.Tensor:
    """
    Compute the gradient of the online loss function with respect to the neural
    network-predicted replacement ray volume properties.

    Parameters 
    ----------
    u
        Fixed zonal wind profile to use during integration.
    model
        Neural network (only used for `build_spectrum`).  
    output
        Replacement ray volume properties as returned by `model`.
    Z
        Correct online pseudomomentum fluxes for each packet.

    Returns
    -------
    torch.Tensor
        Tensor with the same shape as `output` containing the gradient of the
        loss function with respect to each replacement ray property.
    
    """

    if output.requires_grad:
        output = output.detach()

    grad = torch.zeros(output.shape)
    for k in range(output.shape[0]):
        dprop = torch.zeros(output.shape)
        dprop[k] = 0.08 * output[k]

        L_plus = _loss(u, X, output + dprop, Z)
        L_minus = _loss(u, X, output - dprop, Z)

        grad[k] = (L_plus - L_minus) / (2 * dprop[k])

    return grad

def _loss(
    u: torch.Tensor,
    X: torch.Tensor,
    output: torch.Tensor,
    Z: torch.Tensor
) -> torch.Tensor:
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
        Tensor giving the squared error for each wave packet.

    """
    
    wind = torch.vstack((u, torch.zeros_like(u)))
    spectrum = CoarseNet.build_spectrum(X, output)
    Z_hat = integrate_batches(wind, spectrum, 1).mean(dim=1)

    return ((Z - Z_hat) ** 2).sum(dim=1)

