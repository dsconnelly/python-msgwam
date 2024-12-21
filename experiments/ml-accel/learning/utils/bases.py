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
        Three-dimensional tensor, as returned by `transform_proxies`.
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
    amp, shape, shift = proxies[..., None].transpose(0, 1)
    curves = amp * _basis_func(shape * (z - shift), basis_type)

    return torch.nansum(curves, dim=1)

def transform_proxies(
    proxies: torch.Tensor,
    amp_only: bool=False,
    alpha: float=0
) -> torch.Tensor:
    """
    Transform unconstrained proxy variables to appropriately bounded amplitude,
    shape, and shift parameters.

    Parameters
    ----------
    proxies
        Three-dimensional tensor whose first dimension ranges over samples,
        whose second dimension ranges over to the three kinds of parameter, and
        whose third dimension ranges over individual basis function.
    amp_only
        Whether to process all three variable kinds or only the amplitudes. The
        latter is only necessary during coefficient fitting.
    alpha
        The amplitudes are transformed with a leaky ReLU unit, and `alpha` is
        the negative slope of this transformation. Should be zero at inference
        time, but can be varied during coefficient fitting if necessary.

    Returns
    -------
    torch.Tensor
        Tensor of the same shape as `proxies` but with transformed data.

    """

    amp, shape, shift = proxies.transpose(0, 1)
    amp = nn.functional.leaky_relu(amp, alpha)
    amp = amp / amp.sum(dim=1)[:, None]

    if not amp_only:
        shape = nn.functional.softplus(shape)
        shift = 1.1 * _Z_MAX * torch.tanh(shift)

    return torch.stack((amp, shape, shift), dim=1)

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
