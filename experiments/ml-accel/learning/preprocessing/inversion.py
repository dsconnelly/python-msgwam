import numpy as np
import torch, torch.nn as nn

from torch.optim import Adam

from msgwam import config
from msgwam.dispersion import get_m

from ..architectures import load_model, make_inputs
from ..utils import load_data

def invert_surrogte(n_steps: int=100) -> None:
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
        k, m = _unmake_inputs(u, X_hat.detach())
        action_cr = (M / k) ** (1 / 3)

    volume = rays[:, 3:6].prod(dim=-1)
    dens = (action_cr ** 3) / volume

    signs = torch.sign(rays[:, 0])
    data = torch.column_stack((signs * k, m, dens)).numpy()
    np.save(f'data/{config.name}/training/inverted.npy', data)

def _unmake_inputs(
    u: torch.Tensor,
    X_hat: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return data from the neural network input space (phase speed and intrinsic
    period) to wavenumber space.

    Parameters
    ----------
    u
        Tensor of wind profiles, as passed to the neural network.
    X_hat
        Tensor of phase speeds and intrinsic periods, as passed as the first two
        columns of the neural network input.

    Returns
    -------
    torch.Tensor, torch.Tensor
        Arrays of zonal and vertical wavenumbers, respectively.
    
    """

    cp_x, T_hat = X_hat.T
    omega_hat = 2 * torch.pi / T_hat
    k = omega_hat / (cp_x - u[:, 0, 0])
    m = get_m(k, 0, omega_hat, config.N_ref)

    return k, m
