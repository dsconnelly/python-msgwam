import sys

from typing import Optional

import torch

sys.path.insert(0, '.')
from msgwam import config, spectra
from msgwam.dispersion import cg_r
from msgwam.integration import SBDF2Integrator
from msgwam.utils import shapiro_filter

from hyperparameters import smoothing

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
    out: Optional[torch.Tensor]=None,
    smooth: bool=False
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
    out
        Where to store the computed time series. Should be a tensor of shape
        `(PACKETS_PER_BATCH, n_snapshots, n_z)`, where `n_snapshots is the
        number of steps in the time series and `n_z` is the number of grid
        points the wind (and pseudomomentum fluxes) are reported on. If `None`,
        a newly-created array will be returned.
    smooth
        Whether to use Gaussian projection instead of discrete.

    """

    if out is None:
        n_z = config.n_grid - 1
        n_snapshots = config.n_t_max // config.n_skip + 1
        packets_per_batch = spectrum.shape[1] // rays_per_packet
        out = torch.zeros((packets_per_batch, n_snapshots, n_z))

    config.prescribed_wind = wind
    config.n_ray_max = spectrum.shape[1]
    config.n_chromatic = rays_per_packet
    spectra._custom = lambda: spectrum
    config.spectrum_type = 'custom'

    config.refresh()
    solver = SBDF2Integrator().integrate()
    mean = solver.snapshots_mean[0]

    if smooth:
        config.proj_method = 'gaussian'
        config.shapiro_filter = False
        config.smoothing = smoothing
        
    else:
        config.proj_method = 'discrete'
        config.shapiro_filter = True

    config.refresh()
    for j, rays in enumerate(solver.snapshots_rays):
        pmf = (rays.cg_r() * rays.action * rays.k).reshape(-1, rays_per_packet)
        profiles = mean.project(rays, pmf, onto='centers')

        if config.shapiro_filter:
            profiles[1:-1] = shapiro_filter(profiles)

        out[:, j] = profiles.transpose(0, 1)

    return out
