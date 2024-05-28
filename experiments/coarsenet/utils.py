import sys
sys.path.insert(0, '.')

from typing import Optional

import torch, torch.nn as nn

from msgwam import config, spectra
from msgwam.dispersion import cg_r
from msgwam.integration import SBDF2Integrator
from msgwam.utils import shapiro_filter

def get_base_replacement(X: torch.Tensor) -> torch.Tensor:
    """
    Get the base replacement ray volume for each packet. The base replacement
    has the mean properties of all ray volumes in the packet, except for `dr`
    and `dm`, which have the bounding box properties instead.

    Parameters
    ----------
    X
        Tensor containing ray volume properties for each packet, structured
        like a batch in the output of `_sample_wave_packets`.

    Returns
    -------
        Tensor of base replacement ray volumes.

    """

    r_lo = nanmin(X[0] - 0.5 * X[1], dim=-1)
    r_hi = nanmax(X[0] + 0.5 * X[1], dim=-1)

    m_lo = nanmin(X[4] - 0.5 * X[7], dim=-1)
    m_hi = nanmax(X[4] + 0.5 * X[7], dim=-1)

    base = torch.nanmean(X, dim=-1)
    base[1] = r_hi - r_lo
    base[7] = m_hi - m_lo

    return base

def get_batch_pmf(X: torch.Tensor) -> torch.Tensor:
    """
    Compute the pseudomomentum flux of each wave packet in a source spectrum.

    Parameters
    ----------
    X
        Tensor whose first dimension ranges over ray properties, whose second
        dimension ranges over wave packets, and whose third dimension ranges
        over individual ray volumes in each packet. If `X.ndim == 2`, it will be
        assumed that each ray volume constitutes its own wave packet.

    Returns
    -------
    torch.Tensor
        Tensor containing the total absolute pseudomomentum flux in each packet.
    
    """

    if X.ndim == 2:
        X = X[..., None]

    cg = cg_r(*X[2:5])
    volume = X[5:8].prod(dim=0)
    flux = cg * X[-1] * volume * abs(X[2])

    return torch.nansum(flux, dim=1)

def integrate_batches(
    wind: torch.Tensor,
    spectrum: torch.Tensor,
    rays_per_packet: int,
    smoothing: Optional[float]=None
) -> torch.Tensor:
    """
    Integrate the system with the given mean wind profiles (held constant) and
    source spectrum. Then, assume the ray volumes correspond to distinct packets
    and return a time series of zonal pseudomomentum flux profiles for each one.

    Parameters
    ----------
    wind
        Zonal and meridional wind profiles to prescribe during integration.
    spectrum
        Properties of the source spectrum ray volumes.
    rays_per_packet
        How many source ray volumes are in each packet. Used to extract the
        time series for each packet after integration.
    smoothing
        If `None`, discrete projection is used to compute the fluxes. Otherwise,
        sets the smoothing parameter used for Gaussian projection.

    """

    n_z = config.n_grid - 1
    n_snapshots = config.n_t_max // config.n_skip + 1
    packets_per_batch = spectrum.shape[1] // rays_per_packet
    Z = torch.zeros((packets_per_batch, n_snapshots, n_z))

    config.prescribed_wind = wind
    config.n_ray_max = spectrum.shape[1]
    config.n_chromatic = rays_per_packet
    spectra._custom = lambda: spectrum
    config.spectrum_type = 'custom'

    config.refresh()
    solver = SBDF2Integrator().integrate()
    mean = solver.snapshots_mean[0]

    if smoothing is None:
        config.proj_method = 'discrete'
        config.shapiro_filter = True

    else:
        config.proj_method = 'gaussian'
        config.shapiro_filter = False
        config.smoothing = smoothing

    config.refresh()
    for j, rays in enumerate(solver.snapshots_rays):
        pmf = (rays.cg_r() * rays.action * rays.k).reshape(-1, rays_per_packet)
        profiles = mean.project(rays, pmf, onto='centers')

        if config.shapiro_filter:
            profiles[1:-1] = shapiro_filter(profiles)

        Z[:, j] = profiles.transpose(0, 1)

    return Z

def nanmax(a: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Take the maximum over a dimension, ignoring any `torch.nan` values.

    Parameters
    ----------
    a
        Tensor to maximize over.
    dim
        Dimension to maximize over.

    Returns
    -------
    torch.Tensor
        Maximum value in the appropriate dimension, ignoring NaN values.

    """

    return torch.nan_to_num(a, -torch.inf).max(dim=dim)[0]

def nanmin(a: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Take the minimum over a dimension, ignoring any `torch.nan` values.

    Parameters
    ----------
    a
        Tensor to minimize over.
    dim
        Dimension to minimize over.

    Returns
    -------
    torch.Tensor
        Minimum value in the appropriate dimension, ignoring NaN values.

    """

    return torch.nan_to_num(a, torch.inf).min(dim=dim)[0]

def xavier_init(layer: nn.Module):
    """
    Apply Xavier initialization to a layer if it is an `nn.Linear`.

    Parameters
    ----------
    layer
        Module to potentially initialize.

    """

    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight)