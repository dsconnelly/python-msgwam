import numpy as np
import torch, torch.nn as nn

from torch.optim import Adam

from msgwam import config
from msgwam.utils import get_wavenumbers

from ..architectures import load_model, make_inputs
from ..utils import load_data

def invert_surrogate(n_steps: int=100) -> None:
    """
    Invert a coarse surrogate to find new wavenumbers that cause the ray volume
    to behave more like its fine constituents.

    Parameters
    ----------
    n_steps
        How many gradient descent steps to take.

    """

    model, _ = load_model('flux-coarse', 'test', restart=True)
    u, rays, _ = load_data('flux-coarse')
    *_, Y_fine = load_data('flux-fine')

    u, X_hat = make_inputs(u, rays)
    X_hat, action_cr = X_hat[:, :-1].clone(), X_hat[:, -1:]
    M = abs(rays[:, 0]) * action_cr ** 3

    X_hat.requires_grad_(True)
    optimizer = Adam([X_hat], lr=1e-1)
    loss_func = nn.MSELoss()

    for n_step in range(1, n_steps + 1):
        optimizer.zero_grad()
        output = model(u, torch.column_stack(X_hat, action_cr))
        loss = loss_func(output, Y_fine)

        loss.backward()
        optimizer.step()

        print(f'step {n_step}: loss = {loss.item():.6f}')
        k, _ = get_wavenumbers(u, X_hat.detach())
        action_cr = (M / k) ** (1 / 3)

    path = f'data/{config.name}/training/adjustments.npy'
    np.save(path, X_hat.detach().numpy())
