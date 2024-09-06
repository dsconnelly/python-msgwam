from __future__ import annotations
from typing import TYPE_CHECKING, Optional

import torch
import xarray as xr

from msgwam import config, spectra
from msgwam.constants import PROP_NAMES
from msgwam.dispersion import cp_x
from msgwam.integration import SBDF2Integrator
from msgwam.utils import shapiro_filter

if TYPE_CHECKING:
    from msgwam.mean import MeanState
    from msgwam.rays import RayCollection

def integrate_batch(
    wind: torch.Tensor,
    spectrum: torch.Tensor,
    rays_per_packet: int,
    smoothing: Optional[float]=None
) -> torch.Tensor:
    """
    Integrate the system with the given prescribed mean wind profiles and source
    spectrum. Then, assume the ray volumes correspond to distinct packets and
    return a time series of zonal momentum flux profiles for each one.

    Parameters
    ----------
    wind
        Zonal and meridional wind profiles to prescribe during integration.
    spectrum
        Properties of the source spectrum ray volumes.
    rays_per_packet
        How many source ray volumes are in each packet. Also affects breaking.
    smoothing
        If `None`, discrete projection is used to compute the fluxes. Otherwise,
        sets the smoothing parameter used for Gaussian projection.

    Returns
    -------
    torch.Tensor
        Tensor of momentum fluxes whose first dimension ranges over packets in a
        batch, whose second dimension ranges over time steps in the integration,
        and whose third dimension ranges over vertical grid points.

    """

    config.prescribed_wind = wind
    config.n_ray_max = spectrum.shape[1]
    config.n_chromatic = rays_per_packet

    data = {'cp_x' : cp_x(*spectrum[2:5])}
    for i, name in enumerate(PROP_NAMES[:-2]):
        data[name] = ('cp_x', spectrum[i])

    ds = xr.Dataset(data)
    spectra._custom = lambda: ds
    config.spectrum_type = 'custom'

    if smoothing is None:
        config.proj_method = 'discrete'
        config.shapiro_filter = True

    else:
        config.proj_method = 'gaussian'
        config.shapiro_filter = False
        config.smoothing = smoothing

    config.refresh()
    solver = SBDF2Integrator()
    _ = solver.integrate(_get_snapshot)
    config.reset()

    return torch.stack(solver.snapshots).transpose(0, 1)

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

def _get_snapshot(
    mean: MeanState,
    rays: RayCollection
) -> torch.Tensor:
    """
    Calculate the momentum flux profile associated with each packet separately.
    Meant to be passed as the `snapshot_func` argument to the `integrate`
    method of the solver.

    Parameters
    ----------
    mean
        Current mean state of the system.
    rays
        Current ray properties.

    Returns
    -------
    torch.Tensor
        Tensor of shape `(PACKETS_PER_BATCH, config.n_grid - 1)` containing the
        flux profiles at the curent time step.

    """

    pmf = (rays.k * rays.action * rays.cg_r())
    pmf = pmf.reshape(-1, config.n_chromatic)
    profiles = mean.project(rays, pmf, 'centers')

    if config.shapiro_filter:
        profiles[1:-1] = shapiro_filter(profiles)

    return profiles.transpose(0, 1)
