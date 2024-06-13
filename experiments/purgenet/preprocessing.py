import sys

import torch

sys.path.insert(0, '.')
from msgwam import config
from msgwam.integration import SBDF2Integrator
from msgwam.utils import shapiro_filter

PROPS_IN = ['r', 'dr', 'k', 'm', 'dm', 'dens']

WINDOW = 2

def save_training_data() -> None:
    """Generate and save the data necessary to train a `PurgeNet`."""

    u, X, Z = _integrate()

    n_windows = config.n_day // WINDOW
    u = u[1:].reshape(n_windows, -1, u.shape[1])
    X = X[1:].reshape(n_windows, -1, *X.shape[1:])
    Z = Z[1:].reshape(n_windows, -1, *Z.shape[1:])

    X = X[1:, 0].flatten(0, 1)
    Z = Z[1:].mean(dim=1).flatten(0, 1)
    u = u[1:, 0, None].expand(-1, X.shape[1]).flatten(0, 1)
    Y = torch.hstack((u, X))

    torch.save(Y, 'data/purgenet/Y.pkl')
    torch.save(Z, 'data/purgenet/Z.pkl')
    
def _integrate() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Integrate according to the loaded configuration settings, and return mean
    wind profiles, relevant ray volume properties, and momentum flux time series
    for each ray volume.

    Returns
    -------
    torch.Tensor
        Tensor of zonal wind profiles whose first dimension ranges over time
        steps and whose second dimension ranges over vertical grid points.
    torch.Tensor
        Tensor of ray volume properties whose first dimension ranges over time
        steps, whose second dimension ranges over ray volumes, and whose third
        dimension ranges over the properties in `PROPS_IN`.
    torch.Tensor
        Tensor of momentum fluxes whose first dimension ranges over time steps,
        whose second dimension ranges over ray volumes, and whose third
        dimension ranges over vertical grid points.

    """
    
    n_z = config.n_grid - 1
    n_time = config.n_t_max // config.n_skip
    Z = torch.zeros((n_time, config.n_ray_max, n_z))

    solver = SBDF2Integrator().integrate()
    mean = solver.snapshots_mean[0]

    for j, rays in enumerate(solver.snapshots_rays):
        pmf = (rays.k * rays.cg_r() * rays.action).reshape(-1, 1)
        profiles = mean.project(rays, pmf, onto='centers')
        profiles[1:-1] = shapiro_filter(profiles)

        Z[j] = profiles.transpose(0, 1)

    ds = solver.to_dataset()
    u = torch.as_tensor(ds['u'].values)
    X = torch.zeros((n_time, config.n_ray_max, len(PROPS_IN)))

    for i, prop in enumerate(PROPS_IN):
        X[..., i] = torch.as_tensor(ds[prop].values)

    return u, X, Z