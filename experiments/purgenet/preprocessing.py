import sys

import torch

sys.path.insert(0, '.')
from msgwam import config
from msgwam.integration import SBDF2Integrator
from msgwam.utils import open_dataset, shapiro_filter

PROPS_IN = ['r', 'k', 'm', 'dm', 'dens']
N_RANDOMIZATIONS = 6
WINDOW = 0.25

def save_training_data() -> None:
    """Generate and save the data necessary to train a `PurgeNet`."""

    Ys, Zs = [], []
    for _ in range(N_RANDOMIZATIONS):
        u, X, Z = _integrate()
        _randomize_source()

        Y, Z = _process(u, X, Z)
        Ys.append(Y)
        Zs.append(Z)

    torch.save(torch.vstack(Ys), 'data/purgenet/Y.pkl')
    torch.save(torch.vstack(Zs), 'data/purgenet/Z.pkl')

def _randomize_source() -> None:
    """
    Randomizes the properties of the source spectrum, within reasonable
    parameters, and refresh the configuration.
    """

    bounds = {
        'bc_mom_flux' : [1e-3, 5e-3],
        'wvl_hor_char' : [90e3, 110e3],
        'c_center' : [0, 15],
        'c_width' : [8, 16]
    }

    for name, (lo, hi) in bounds.items():
        sample = (hi - lo) * torch.rand(1) + lo
        setattr(config, name, sample.item())

    config.refresh()

def _process(
    u: torch.Tensor,
    X: torch.Tensor,
    Z: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert integration outputs into training inputs and targets.

    Parameters
    ----------
    u
        Tensor of zonal wind profiles as returned by `_integrate`.
    X
        Tensor of ray volume properties as returned by `_integrate`.
    Z
        Tensor of momentum flux time series as returned by `_integrate`.

    Returns
    -------
    torch.Tensor
        Tensor of training inputs whose columns correspond first to the zonal
        wind profile and then to ray volume properties.
    torch.Tensor
        Tensor of training targets whose columns correspond to mean momentum
        fluxes at each vertical grid level.

    """

    n_windows = int(config.n_day // WINDOW)
    u = u[1:].reshape(n_windows, -1, u.shape[1])[1:]
    X = X[1:].reshape(n_windows, -1, *X.shape[1:])[1:]
    Z = Z[1:].reshape(n_windows, -1, *Z.shape[1:])[1:]

    X, meta = X[..., :-1], X[..., -1]
    keep = meta == meta[:, 0:1]
    Z[~keep] = 0

    X = X[:, 0]
    Z = Z[:].mean(dim=1)
    u = u[:, 0, None].expand(-1, X.shape[1], -1)

    u = u.flatten(0, 1)
    X = X.flatten(0, 1)
    Y = torch.hstack((u, X))
    Z = Z.flatten(0, 1)

    return Y, Z
    
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
        dimension ranges over the properties in `PROPS_IN`. Also returns `meta`,
        so that we can later exclude rays that are purged in given window.
    torch.Tensor
        Tensor of momentum fluxes whose first dimension ranges over time steps,
        whose second dimension ranges over ray volumes, and whose third
        dimension ranges over vertical grid points.

    """
    
    n_z = config.n_grid - 1
    n_time = config.n_t_max // config.n_skip
    n_time = n_time + (config.dt_output > config.dt)
    Z = torch.zeros((n_time, config.n_ray_max, n_z))

    with open_dataset(config.prescribed_wind) as ds:
        last = len(ds['time']) - config.n_t_max
        start = torch.randint(last, size=(1,)).item()
        ds = ds.isel(time=slice(start, start + config.n_t_max))

        u = torch.as_tensor(ds['u'].values)
        v = torch.as_tensor(ds['v'].values)
        wind = torch.stack((u, v), dim=1)
        config.prescribed_wind = wind

    config.refresh()
    solver = SBDF2Integrator().integrate()
    mean = solver.snapshots_mean[0]
    config.reset()

    for j, rays in enumerate(solver.snapshots_rays):
        pmf = (rays.k * rays.cg_r() * rays.action).reshape(-1, 1)
        profiles = mean.project(rays, pmf, onto='centers')
        profiles[1:-1] = shapiro_filter(profiles)

        Z[j] = profiles.transpose(0, 1)

    ds = solver.to_dataset()
    u = torch.as_tensor(ds['u'].values)
    X = torch.zeros((n_time, config.n_ray_max, len(PROPS_IN) + 1))

    for i, prop in enumerate(PROPS_IN + ['meta']):
        X[..., i] = torch.as_tensor(ds[prop].values)

    return u, X, Z