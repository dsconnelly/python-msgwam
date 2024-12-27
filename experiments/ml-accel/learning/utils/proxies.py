from typing import Optional
from warnings import catch_warnings

import torch, torch.nn as nn

from msgwam import config

from .. import hyperparameters as hp
from .statistics import nanstd

_Z_MAX = 0.5

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

def get_proxy_statistics(
    proxies: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate the correct mean and standard deviation for unconstrained proxy
    variables. The main complication is that the amplitudes should be treated as
    though the mean is zero, to avoid bias from the ReLU transformation.

    Parameters
    ----------
    proxies
        Tensor of proxy variables, as passed to `transform_proxies`.

    Returns
    -------
    torch.Tensor, torch.Tensor
        Tensors of means and standard deviations, each two-dimensional with the
        first dimension ranging over the three kinds of proxy variables, and the
        second dimension ranging over individual basis functions.

    """

    amp, *_ = proxies.transpose(0, 1)
    with catch_warnings(action='ignore', category=RuntimeWarning):
        means = torch.nanmean(proxies, dim=0)
        stds = nanstd(proxies, dim=0)
        
    means[0] = 0
    amp = amp.clone()
    amp[amp == 0] = torch.nan
    # stds[0] = torch.sqrt(torch.nanmean(amp ** 2, dim=0))
    stds[0] = nanstd(amp, dim=0)

    return torch.nan_to_num(means), torch.nan_to_num(stds)

def init_proxies(n_packets: int) -> torch.Tensor:
    """
    Generate a good initial guess for the (unconstrained) proxies that is likely
    to allow fitting to converge faster.

    Parameters
    ----------
    n_packets
        Number of packets to generate guesses for.

    Returns
    -------
    torch.Tensor
        Tensor of proxy variables, as passed to `transform_proxies`. Has
        `requires_grad` set to `True`.

    """

    amp = torch.ones(hp.n_basis) / hp.n_basis
    shape = torch.log(20 * torch.ones(hp.n_basis))
    shift = torch.atanh(torch.linspace(-_Z_MAX, _Z_MAX, hp.n_basis))

    proxies = torch.vstack((amp, shape, shift)).double()
    proxies = proxies[None].expand(n_packets, -1, -1).clone()
    proxies.requires_grad_(True)

    return proxies

def transform_proxies(
    proxies: torch.Tensor,
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
    amp = amp / torch.clamp(amp.sum(dim=1)[:, None], min=1e-12)

    shape = torch.exp(shape)
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
