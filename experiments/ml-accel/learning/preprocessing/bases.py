from time import time

import numpy as np
import torch, torch.nn as nn

from msgwam import config

from .. import hyperparameters as hp
from ..utils import (
    add_task_info,
    apply_basis,
    get_overrides,
    get_workload,
    load_data
)

def save_basis_coefficients(
    grain: str,
    max_hours: int=5,
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
    start, end = get_workload(Y.shape[0])
    shape = (end - start, 3 * hp.n_basis)
    Y = Y[start:end]

    coeffs = torch.rand(*shape, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([coeffs], lr=0.1)
    loss_func = nn.MSELoss()

    n_step, start = 1, time()
    min_loss, waited = torch.inf, 0

    with config.override(n_grid=get_overrides()['n_grid']):
        while n_step < max_steps + 1 and (time() - start) / 3600 < max_hours:
            optimizer.zero_grad()

            output = apply_basis(coeffs)
            loss = loss_func(output, Y)

            loss.backward()
            optimizer.step()
            print(f'step {n_step}: loss = {loss.item():.6f}')

            waited += 1
            if loss < min_loss:
                min_loss = loss
                waited = 0

            if waited > 100:
                print('patience exceeded!')
                break

            if loss < stop_loss:
                print('terminating early!')
                break

            n_step = n_step + 1

    coeffs = coeffs.detach().numpy()
    fname = f'coeffs-{grain}-{hp.basis_type}.npy'
    path = add_task_info(f'data/{config.name}/training/{fname}')
    np.save(path, coeffs)
