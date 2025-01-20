from typing import Optional

import torch, torch.nn as nn

from msgwam import config
from msgwam.dispersion import get_omega_hat

def make_inputs(
    u: torch.Tensor,
    rays: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Preprocess input data to be passed to a `SourceNet`. Extracts spectral
    features from ray volume data, and handles the sign of the zonal wind.

    Parameters
    ----------
    u
        Tensor of zonal wind profiles.
    rays
        Tensor of ray volume properties.

    Returns
    -------
    torch.Tensor
        Tensor of sign-modified zonal wind data.
    torch.Tensor
        Tensor of extracted spectral properties.

    """

    k, l, m, dk, dl, dm, dens = rays.T
    action = (dens * dk * dl * dm) ** (1 / 3)
    omega_hat = get_omega_hat(k, l, m, config.N_ref)

    T_hat = 2 * torch.pi / omega_hat
    cp_x = torch.sign(k) * (omega_hat / k + u[:, 0, 0])
    u = u * torch.sign(k)[:, None, None]

    return u, torch.column_stack((cp_x, T_hat, action))

def standardize(
    a: torch.Tensor,
    means: Optional[torch.Tensor]=None,
    stds: Optional[torch.Tensor]=None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Standardize a tensor along the first dimension.

    Parameters
    ----------
    a
        Tensor containing data to standardize.
    means
        Means to use during standardization. If `None`, the mean along the first
        dimension will be computed and used.
    stds
        Standard deviations to use during standardization. If `None`, the
        standard deviation along the first dimension will be computed and used.

    Returns
    -------
    torch.Tensor
        Standardized data.
    torch.Tensor, torch.Tensor
        Means and standard deviations used during standardization. If either of
        these statistics was provided, they will be returned as is.

    """

    if means is None:
        means = a.mean(dim=0)

    if stds is None:
        stds = a.std(dim=0)

    sdx = stds > 0
    output = torch.zeros_like(a)
    output[:, sdx] = (a - means)[:, sdx] / stds[sdx]

    return output, means, stds

def xavier_init(a: nn.Module | torch.Tensor) -> None:
    """
    Apply Xavier initialization a linear layer or a weight matrix.

    Parameters
    ----------
    a
        Module or weight matrix to potentially initialize. If an `nn.Linear` is
        passed, its weight matrix will be extracted and initialized.

    """

    if isinstance(a, (nn.Linear, nn.Conv1d)):
        a = a.weight

    if isinstance(a, torch.Tensor):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(a, gain=gain)
