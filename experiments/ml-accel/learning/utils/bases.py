from typing import Optional

import torch, torch.nn as nn

from msgwam import config

from .. import hyperparameters as hp

def apply_basis(
    coeffs: torch.Tensor,
    n_grid: Optional[int]=None
) -> torch.Tensor:
    """
    Given a tensor of amplitude, shape, and shift parameters, compute the
    profile given as the sum of the basis functions for each sample.

    Parameters
    ----------
    coeffs
        Tensor whose first dimension ranges over training samples and whose
        second dimension ranges over coefficients for the basis functions.
    n_grid
        Number of points in the coordinate grid on which to evaluate the basis
        functions. If `None`, uses the the value set in `config`.

    Returns
    -------
    torch.Tensor
        Profile corresponding to each sample.

    """

    if n_grid is None:
        n_grid = config.n_grid

    n_samples = coeffs.shape[0]
    coeffs = coeffs.reshape(n_samples, 3, -1, 1)
    amp, shape, shift = coeffs.transpose(0, 1)
    z = -torch.linspace(-3, 3, n_grid)

    amp = torch.softmax(amp, dim=1)
    shape = nn.functional.softplus(shape)
    shift = 1.1 * z.max() * torch.tanh(shift)

    arg = shape * (z - shift)
    curves = amp * _basis_func(arg)

    return curves.sum(dim=1)

def _basis_func(z: torch.Tensor) -> torch.Tensor:
    """
    Compute the normalized version of the basis function, which must have
    unit slope at the origin and be bounded between zero and one.

    Parameters
    ----------
    z
        Tensor of input values.

    Returns
    -------
    torch.Tensor
        Basis function values.

    """

    if hp.basis_type == 'logistic':
        return 1 / (1 + torch.exp(-4 * z))
    
    if hp.basis_type == 'quadratic':
        return (1 + 2 * z / torch.sqrt(1 + (2 * z) ** 2)) / 2
    
    if hp.basis_type == 'tanh':
        return (1 + torch.tanh(2 * z)) / 2

    raise ValueError(f'Unknown basis type: {hp.basis_type}')
