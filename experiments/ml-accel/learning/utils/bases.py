from typing import Optional

import torch, torch.nn as nn

from msgwam import config

from .. import hyperparameters as hp

_Z_MAX = 3

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

    z = -torch.linspace(-_Z_MAX, _Z_MAX, n_grid)
    amp, shape, shift = parse_proxies(proxies, add_z_dim=True)
    curves = amp * _basis_func(shape * (z - shift), basis_type)

    return curves.sum(dim=1)

def parse_proxies(
    proxies: torch.Tensor,
    add_z_dim: bool=False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Unpack a two-dimensional tensor of proxy variables into amplitude, shape,
    and shift parameters, performing the necessary transformations.

    Parameters
    ----------
    proxies
        Tensor of proxy variables, as passed to `apply_basis`.
    add_z_dim
        Whether to append a dummy dimension so that these parameters can be used
        later to evaluate the basis functions on a vertical grid.

    Returns
    -------
    torch.Tensor, torch.Tensor, torch.Tensor
        Two-dimensional tensors of ampltiude, shape, and shift parameters.

    """

    proxies = proxies.reshape(proxies.shape[0], 3, -1)
    
    if add_z_dim:
        proxies = proxies[..., None]

    amp, shape, shift = proxies.transpose(0, 1)

    amp = torch.softmax(amp, dim=1)
    shape = nn.functional.softplus(shape)
    shift = 1.1 * _Z_MAX * torch.tanh(shift)

    return amp, shape, shift

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
