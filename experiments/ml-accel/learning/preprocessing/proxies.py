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
    init_proxies,
    load_data,
    transform_proxies
)

def save_proxies(
    grain: str,
    basis_type: str='logistic',
    max_steps: int=20000,
    max_hours: int=5,
    rolloff_start: int=1500,
    rolloff_end: int=2500,
    patience: int=600,
) -> None:
    """
    Compute the best representation of the momentum flux profiles with a given
    basis family, so that these coefficients can be learned later.

    Parameters
    ----------
    grain
        Whether to work with `'coarse'` or `'fine'` packets.
    max_steps
        How many steps to take before terminating.
    max_hours
        How long the optimization can run before terminating.
    rolloff_start, rolloff_end
        Fitting proceeds in three stages. First, the amplitudes are normalized
        with a leaky ReLU with constant negative slope. Next, the negative slope
        is rolled off (decreased linearly to zero). Finally, fitting continues
        with a hard ReLU until the loss fails to decrease for sufficiently many
        steps. These two arguments set the start and end of the rolloff phase.    
    patience
        How many steps can occur without lowering the loss before termination
        after the negative slope has reached zero.

    """

    Y = abs(load_data(f'flux-{grain}')[-1])
    start, end = get_workload(Y.shape[0])
    shape = (end - start, 3, hp.n_basis)
    Y = torch.clamp(Y[start:end], max=1)

    proxies = init_proxies(end - start)
    optimizer = torch.optim.Adam([proxies], lr=0.01)
    loss_func = nn.MSELoss()

    n_step, start = 1, time()
    min_loss, n_stuck = torch.inf, 0
    best_proxies = torch.zeros_like(proxies)

    alpha = 1e-2
    decrement = alpha / (rolloff_end - rolloff_start)

    with config.override(n_grid=get_overrides()['n_grid']):
        while n_step < max_steps + 1 and ((time() - start) / 3600) < max_hours:
            optimizer.zero_grad()

            post = transform_proxies(proxies, alpha=alpha)
            output = apply_basis(post, basis_type=basis_type)
            loss = loss_func(output, Y)

            loss.backward()
            optimizer.step()
            print(f'step {n_step}: loss = {loss.item():.8f}')

            if rolloff_start <= n_step < rolloff_end:
                alpha = max(alpha - decrement, 0)

            if rolloff_end <= n_step:
                n_stuck = n_stuck + 1

                if loss < min_loss:
                    best_proxies = proxies.clone()
                    min_loss = loss
                    n_stuck = 0

                if n_stuck > patience:
                    print('Patience exceeded, terminating early')
                    break

            n_step = n_step + 1

    print(f'Best loss was {min_loss:.8f}')
    proxies = transform_proxies(best_proxies.detach())
    amp, shape, shift = proxies.transpose(0, 1)

    idx = amp == 0
    shape[idx] = torch.nan
    shift[idx] = torch.nan

    jdx = torch.argsort(shift, dim=1)[:, None]
    proxies = torch.take_along_dim(proxies, jdx, dim=2)

    fname = f'proxies-{grain}-{basis_type}.npy'
    path = add_task_info(f'data/{config.name}/training/{fname}')
    np.save(path, proxies.numpy())
