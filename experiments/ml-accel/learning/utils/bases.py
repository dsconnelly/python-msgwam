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
        Three-dimensional tensor, as returned by `postprocess_proxies`.
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

    return curves.sum(dim=1)

def postprocess_proxies(
    proxies: torch.Tensor,
    training: bool=False
) -> torch.Tensor:
    """
    Transform unconstrained data into amplitude, shape, and shift parameters.

    Parameters
    ----------
    proxies
        Three-dimensional tensor whose first dimension ranges over samples,
        whose second dimension ranges over the three kinds of proxy variable,
        and whose third dimension ranges over basis functions.
    training
        If `True`, then the amplitudes will be set to positive using a leaky
        ReLU, and the other parameters will not be zeroed out, to facilitate
        better neural network training.

    Returns
    -------
    torch.Tensor
        Postprocessed tensor of the same shape as `proxies`. Variables are
        constrained to fall within sensible bounds, and the shape and shift
        variables are set to zero where the amplitude is zero.

    """

    amp, shape, shift = proxies.transpose(0, 1)
    func = nn.functional.leaky_relu if training else nn.functional.relu

    amp = func(amp)
    amp = amp / amp.sum(dim=1)[:, None]
    shape = nn.functional.softplus(shape)
    shift = 1.1 * _Z_MAX * torch.tanh(shift)

    if not training:
        idx = amp == 0
        shape[idx] = 0
        shift[idx] = 0

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
