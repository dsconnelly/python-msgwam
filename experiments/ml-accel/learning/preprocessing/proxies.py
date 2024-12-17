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

def save_proxies(
    grain: str,
    basis_type: str='logistic',
    max_hours: int=5,
    max_steps: int=5000,
    patience: int=100,
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
    patience
        How many steps can occur without lowering the loss before termination.
    stop_loss
        Loss value below which the optimization will terminate early.

    """

    Y = abs(load_data(f'flux-{grain}')[-1])
    start, end = get_workload(Y.shape[0])
    shape = (end - start, 3 * hp.n_basis)
    Y = Y[start:end]

    proxies = torch.rand(*shape, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([proxies], lr=0.1)
    loss_func = nn.MSELoss()

    n_step, start = 1, time()
    min_loss, n_stuck = torch.inf, 0

    with config.override(n_grid=get_overrides()['n_grid']):
        while n_step < max_steps + 1 and ((time() - start) / 3600) < max_hours:
            optimizer.zero_grad()

            output = apply_basis(proxies, basis_type=basis_type)
            loss = loss_func(output, Y)

            loss.backward()
            optimizer.step()
            print(f'step {n_step}: loss = {loss.item():.6f}')

            n_stuck += 1
            if loss < min_loss:
                min_loss = loss
                n_stuck = 0

            if n_stuck > patience:
                print('Patience exceeded, terminating early')
                break

            if loss < stop_loss:
                print('Stop loss achieved, terminating early')
                break

            n_step = n_step + 1

    proxies = proxies.detach().numpy()
    fname = f'proxies-{grain}-{basis_type}.npy'
    path = add_task_info(f'data/{config.name}/training/{fname}')
    np.save(path, proxies)
