import numpy as np
import torch, torch.nn as nn

from torch.optim import Adam

from msgwam import config
from msgwam.utils import get_wavenumbers

from ... import hyperparameters as hp

from ..architectures import load_model, make_inputs
from ..utils import add_task_info, get_overrides, get_workload, load_data

from .generation import _generate_outputs

def invert_surrogate(n_steps: int=500) -> None:
    """
    Invert a coarse surrogate to find new wavenumbers that cause the ray volume
    to behave more like its fine constituents.

    Parameters
    ----------
    n_steps
        How many gradient descent steps to take.

    """

    with config.override(n_grid=get_overrides()['n_grid']):
        model, _ = load_model('flux-coarse', 'test', restart=True)
        model.eval()

    u, rays, Y_coarse = load_data('flux-coarse', distributed=True)
    *_, Y_fine = load_data('flux-fine', distributed=True)
    Y_fine, Y_coarse = abs(Y_fine), abs(Y_coarse)

    u, rays = u[:1000], rays[:1000]
    Y_coarse = Y_coarse[:1000]
    Y_fine = Y_fine[:1000]

    u, X_hat = make_inputs(u, rays)
    X_hat, action_cr = X_hat[:, :-1].clone(), X_hat[:, -1]
    M = abs(rays[:, 0]) * action_cr ** 3

    X_hat.requires_grad_(True)
    optimizer = Adam([X_hat], lr=1e-1)
    loss_func = nn.MSELoss()

    best_X_hat = None
    best_loss = torch.inf

    for n_step in range(1, n_steps + 1):
        optimizer.zero_grad()
        output = model(u, torch.column_stack((X_hat, action_cr)))
        loss = loss_func(output, Y_fine)

        loss.backward()
        optimizer.step()

        print(f'step {n_step}: loss = {loss.item():.6f}')
        k, _ = get_wavenumbers(u, X_hat.detach())
        action_cr = abs(M / k) ** (1 / 3)

        if loss.item() < best_loss:
            best_X_hat = X_hat.detach()
            best_loss = loss

    X_hat = best_X_hat
    print(f'Best loss was {best_loss:.6f}')

    path = f'data/{config.name}/training/candidates.npy'
    np.save(add_task_info(path), X_hat.detach().numpy())

def validate_inversion() -> None:
    """
    Validate the inversion by integrating with the adjusted wavenumbers and
    keeping only those adjustments that improve errors relative to fine.
    """

    kwargs = get_overrides(fine=False)
    path = f'data/{config.name}/training/candidates.npy'
    kwargs['network_path'] = add_task_info(path)
    kwargs['source_type'] = 'network'

    with config.override(**kwargs):
        a, b = get_workload(hp.generation.n_packets)
        Y_hat = _generate_outputs(b - a)
    
    *_, Y_coarse = load_data('flux-coarse', distributed=True)
    *_, Y_fine = load_data('flux-fine', distributed=True)

    errors_coarse = ((Y_coarse - Y_fine) ** 2).sum(dim=1)
    errors_hat = ((Y_hat - Y_fine) ** 2).sum(dim=1)
    keep = errors_hat < errors_coarse

    data = np.load(kwargs['network_path'])
    path = f'data/{config.name}/training/adjustments.npy'
    np.save(add_task_info(path), data[keep])