from time import time

import numpy as np
import torch, torch.nn as nn

from msgwam import config

from .. import hyperparameters as hp
from ..utils import apply_basis, get_overrides, load_data

def save_basis_coefficients(
    grain: str,
    max_hours: int=23,
    max_steps: int=5000,
    stop_loss: float=0.00001
) -> None:
    """
    Compute the best representation of the momentum flux profiles with a given
    basis family, so that these coefficients can be learned later.

    Parameters
    ----------
    grain
        Whether to work with `'coarse'` or `'fine'` packets.
    max_hours
        How long the optimization can run before terminating.
    max_steps
        How many steps to take before terminating.
    stop_loss
        Loss value below which the optimization will terminate early.

    """

    Y = abs(load_data('flux', grain)[-1])
    shape = (Y.shape[0], 3 * hp.n_basis)

    coeffs = torch.rand(*shape, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([coeffs], lr=0.1)
    loss_func = nn.MSELoss()

    n_step, start = 1, time()
    with config.override(n_grid=get_overrides()['n_grid']):
        while n_step < max_steps + 1 and (time() - start) / 3600 < max_hours:
            optimizer.zero_grad()

            output = apply_basis(coeffs)
            loss = loss_func(output, Y)

            loss.backward()
            optimizer.step()
            print(f'step {n_step + 1}: loss = {loss.item():.6f}')

            if loss < stop_loss:
                print('terminating early!')
                break

    coeffs = coeffs.detach().numpy()
    fname = f'coeffs-{grain}-{hp.basis_type}.npy'
    np.save(f'data/{config.name}/training/{fname}', coeffs)
