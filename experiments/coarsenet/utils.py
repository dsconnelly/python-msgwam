import sys

import torch

sys.path.insert(0, '.')
from msgwam import config, spectra
from msgwam.integration import SBDF2Integrator
from msgwam.utils import get_fracs, shapiro_filter

def batch_integrate(
    wind: torch.Tensor,
    spectrum: torch.Tensor,
    rays_per_packet: int,
    out: torch.Tensor
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
        `(n_snapshots, packets_per_batch, n_wind)`, where `n_snapshots` is the
        number of steps in the time series and `n_wind` is the number of grid
        points the wind (and pseudomomentum fluxes) are reported on.

    """

    config.prescribed_wind = wind
    config.n_ray_max = spectrum.shape[1]
    config.n_chromatic = rays_per_packet

    config.spectrum_type = 'custom'
    spectra._custom = lambda: spectrum
    config.refresh()

    solver = SBDF2Integrator().integrate()
    faces = solver.snapshots_mean[0].z_faces
    shape = (wind.shape[1], -1, rays_per_packet)

    for j, rays in enumerate(solver.snapshots_rays):
        pmf = get_fracs(rays, faces) * rays.cg_r() * rays.action * rays.k
        profiles = torch.nansum(pmf.reshape(shape), dim=2)
        profiles[1:-1] = shapiro_filter(profiles)
        out[j] = profiles.transpose(0, 1)
