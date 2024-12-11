from __future__ import annotations
from typing import TYPE_CHECKING

import torch

from torch.optim import Adam

from msgwam import config

from .utils import load_data, load_model

if TYPE_CHECKING:
    from .architectures import SourceNet

def invert_surrogate(
    n_steps: int=100,
    n_print: int=1
) -> None:
    """
    Invert a pretrained coarse `Surrogate` to find ray volumes that should yield
    online flux profiles more similar to those associated with the underlying
    fine packets than the naive coarsening. Save the revised spectrum as a
    netCDF file for later validation.
    """

    model, _ = load_model('coarse', 'test', restart=True)
    u, X, Y_coarse = load_data('coarse')
    *_, Y_fine = load_data('fine')

    n_z = config.n_grid - 1
    stacked = model._preprocess(u, X)
    stacked = model._standardize(stacked)
    u, X_hat = stacked[:, :n_z], stacked[:, n_z:]

    X_hat.requires_grad_(True)
    optimizer = Adam([X_hat], lr=1e-1)

    for n_step in range(1, n_steps + 1):
        optimizer.zero_grad()
        output = _forward(u, X, X_hat, model)
        loss = _loss_func(Y_fine, output).mean()

        loss.backward()
        optimizer.step()

        if n_step % n_print == 0:
            print(f'step {n_step}: loss = {loss.item():.6g}')

def _forward(
    u: torch.Tensor,
    X: torch.Tensor,
    X_hat: torch.Tensor,
    model: SourceNet
) -> torch.Tensor:
    """
    
    """

    stacked = torch.hstack((u, X_hat))
    output = model._predict(stacked)

    return model._postprocess(u, X, output)

def _loss_func(target: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
    """
    
    """

    return ((output - target) ** 2).sum(dim=-1)
