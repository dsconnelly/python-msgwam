from __future__ import annotations
from os.path import dirname as parent
from typing import TYPE_CHECKING, Optional

import torch, torch.nn as nn
import xarray as xr

from msgwam import config, spectra
from msgwam.constants import PROP_NAMES
from msgwam.dispersion import cp_x
from msgwam.integration import SBDF2Integrator
from msgwam.utils import shapiro_filter

if TYPE_CHECKING:
    from msgwam.mean import MeanState
    from msgwam.rays import RayCollection

_REPO_DIR = parent(parent(parent(__file__)))
DATA_DIR = f'{_REPO_DIR}/data/ml-accel'
LOG_DIR = f'{_REPO_DIR}/logs/ml-accel'

def dimensionalize(
    Y: torch.Tensor,
    Z: torch.Tensor,
    T: Optional[float]=None
) -> torch.Tensor:
    """
    Transform neural network outputs to fluxes in physical units.

    Parameters
    ----------
    Y
        Coarse ray volume properties as to be passed to `_get_flux_scale`.
    Z
        Dimensionless flux profiles (with amplitudes close to unity) as
        output by `forward` of a `SurrogateNet`.
    T
        Time horizon over which the profile should be interpreted as an average.

    Returns
    -------
    torch.Tensor
        Two-dimensional tensor of flux profiles with physical units whose
        second dimension ranges over vertical grid points.

    """

    return Z * _get_flux_scale(Y, T)[:, None]

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

def load_data(
    target_type: str,
    nondimensional: bool=True
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load input and output data and perform some useful reshaping.

    Parameters
    ----------
    target_type
        Whether to load momentum flux profiles associated with `'fine'` or
        `'coarse'` ray volumes, or ray volume properties `'Y_hat'` obtained by
        inverting a trained surrogate model.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        Triples of two-dimensional (u, Y, target) target tensors.
            
    """

    u = torch.load(f'{DATA_DIR}/wind.pkl')[:, 0, 0]
    X = torch.load(f'{DATA_DIR}/X.pkl').transpose(1, 2)
    Y = torch.load(f'{DATA_DIR}/Y.pkl').transpose(1, 2)

    u = u[:, None].expand(-1, Y.shape[1], -1).flatten(0, 1)
    X, Y = X.flatten(0, 1), Y.flatten(0, 1)

    if target_type in ['fine', 'coarse']:
        targets = torch.load(f'{DATA_DIR}/Z-{target_type}.pkl').flatten(0, 1)

        if nondimensional:
            spectrum = {'fine' : X, 'coarse' : Y}[target_type]
            targets = nondimensionalize(spectrum, targets)

    elif target_type == 'Y-hat':        
        mask = torch.load(f'{DATA_DIR}/mask-candidates.pkl')
        Y_hat = torch.load(f'{DATA_DIR}/Y-hat.pkl').transpose(1, 2)
        mask, Y_hat = mask.flatten(), Y_hat.flatten(0, 1)

        idx_new = torch.nonzero(mask).flatten()
        idx_old = torch.nonzero(~mask).flatten()
        idx_old = idx_old[torch.randperm(len(idx_old))[:len(idx_new)]]
        idx_all = torch.cat((idx_new, idx_old))

        # targets = torch.vstack((Y_hat[idx_new], Y[idx_old]))
        # u, Y = u[idx_all], Y[idx_all]

        targets = Y_hat[idx_new]
        u, Y = u[idx_new], Y[idx_new]

    else:
        raise ValueError(f'Unknown target type: {target_type}')
    
    return u, Y, targets

def nondimensionalize(Y: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
    """
    Transform physical data to profiles with amplitudes close to unity more
    suitable for training and comparing neural networks.

    Parameters
    ----------
    Y
        Coarse ray volume properties as to be passed to `_get_flux_scale`.
    Z
        Flux profiles with physical units.

    Returns
    -------
    torch.Tensor
        Two-dimensional tensor of nondimensional flux profiles second
        dimension ranges over vertical grid points.

    """

    return Z / _get_flux_scale(Y)[:, None]

def xavier_init(layer: nn.Module) -> None:
    """
    Apply Xavier initialization to a layer if it is an `nn.Linear`.

    Parameters
    ----------
    layer
        Module to potentially initialize.

    """

    if isinstance(layer, nn.Linear):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(layer.weight, gain=gain)

def _get_flux_scale(
    spectrum: torch.Tensor,
    T: Optional[float]=None
) -> torch.Tensor:
    """
    Compute the scale factors for each profile used to (non)dimensionalize.

    Parameters
    ----------
    spectrum
        Two- or three-dimensional tensor of coarse ray volume properties. The
        first dimension should range over batches, and the second dimension
        should range over ray volume properties. The third dimension, if it is
        present, should range over rays within each batch.
    T
        Time horizon over which the profile should be interpreted as an average.
        If `None`, uses the integration length in the current config file.

    Returns
    -------
    torch.Tensor
        One-dimensional tensor of factors, one for each in row in `spectrum`,
        that scale transform the corresponding flux profiles to have amplitudes
        close to unity. Some functional forms might work better than others.

    """

    if T is None:
        T = config.n_t_max * config.dt

    _, dr, k, *_, dk, dl, dm, dens = spectrum.transpose(0, 1)
    scale = abs(k) * (dens * dk * dl * dm) * dr / T

    if scale.dim() > 1:
        scale = scale.sum(dim=1)

    return scale

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
    profiles = mean.project(rays, pmf, 'faces')

    if config.shapiro_filter:
        profiles[1:-1] = shapiro_filter(profiles)

    return profiles.transpose(0, 1)
