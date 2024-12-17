from typing import Optional

import torch, torch.nn as nn

from msgwam import config

from .. import hyperparameters as hp

def apply_basis(
    proxies: torch.Tensor,
    n_grid: Optional[int]=None,
    basis_type: Optional[str]=None
) -> torch.Tensor:
    """
    Given a tensor of amplitude, shape, and shift parameters, compute the
    profile given as the sum of the basis functions for each sample.

    Parameters
    ----------
    proxies
        Tensor whose first dimension ranges over training samples and whose
        second dimension ranges over coefficients for the basis functions.
    n_grid
        Number of points in the coordinate grid on which to evaluate the basis
        functions. If `None`, uses the the value set in `config`.
    basis_type
        What family of basis functions to use. Defaults to the value set by
        the current hyperparameter configuration.

    Returns
    -------
    torch.Tensor
        Profile corresponding to each sample.

    """

    if n_grid is None:
        n_grid = config.n_grid

    if basis_type is None:
        basis_type = hp.basis_type

    n_samples = proxies.shape[0]
    proxies = proxies.reshape(n_samples, 3, -1, 1)
    amp, shape, shift = proxies.transpose(0, 1)
    z = -torch.linspace(-3, 3, n_grid)

    amp = torch.softmax(amp, dim=1)
    shape = nn.functional.softplus(shape)
    shift = 1.1 * z.max() * torch.tanh(shift)

    arg = shape * (z - shift)
    curves = amp * _basis_func(arg)

    return curves.sum(dim=1)

def _basis_func(z: torch.Tensor, basis_type: str) -> torch.Tensor:
    """
    Compute the normalized version of the basis function, which must have
    unit slope at the origin and be bounded between zero and one.

    Parameters
    ----------
    z
        Tensor of input values.
    basis_type
        What family of basis functions to use.

    Returns
    -------
    torch.Tensor
        Basis function values.

    """

    if basis_type == 'logistic':
        return 1 / (1 + torch.exp(-4 * z))
    
    if basis_type == 'quadratic':
        return (1 + 2 * z / torch.sqrt(1 + (2 * z) ** 2)) / 2

    raise ValueError(f'Unknown basis type: {basis_type}')
