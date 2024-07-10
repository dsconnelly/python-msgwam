import sys
sys.path.insert(0, '.')

from typing import Optional

import torch, torch.nn as nn

from msgwam import config, spectra
from msgwam.integration import SBDF2Integrator
from msgwam.utils import shapiro_filter

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

    Returns
    -------
    torch.Tensor
        Tensor of momentum fluxes whose first dimension ranges over packets in a
        batch, whose second dimension ranges over time steps within an
        integration, and whose third dimension ranges over vertical grid points.

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

def root_transform(a: torch.Tensor, root: int) -> torch.Tensor:
    """
    Transform data by taking a sign-aware root.

    Parameters
    ----------
    a
        Data to be transformed.
    root
        Order of the root to take. For example, `root=3` takes a cube root.

    Returns
    -------
    torch.Tensor
        Transformed data.

    """

    return torch.sign(a) * (abs(a) ** (1 / root))

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