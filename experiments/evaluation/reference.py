import cftime
import numpy as np
import xarray as xr

from msgwam import config
from msgwam.constants import EPOCH
from msgwam.integration import integrate
from msgwam.means import InteractiveWind
from msgwam.plotting import plot_boundary, plot_integration, plot_ray_count
from msgwam.utils import make_colored_noise, shapiro_filter

from utils import get_min_dr

_OVERRIDES = {
    'dt' : 10,
    'n_grid' : 201,
    'n_source' : 200,
    'n_max' : 50000,
    'n_increment' : 5000,
    'prune_by' : 'none'
}

def save_mean_state(seed: int=1234) -> None:
    """
    Save the mean wind to be used for this configuration. The wind is saved at
    the reference resolution, from which it can be coarsened later.

    Parameters
    ----------
    seed
        Integer to use as the random seed.

    """

    np.random.seed(seed)
    with config.override(**_OVERRIDES):
        _get_descending_jets().to_netcdf(f'data/{config.name}/mean-state.nc')

def save_reference() -> None:
    """
    Save the reference integration to compute errors against. The reference run
    has no pruning and extremely high values for the time step and the number of
    spectral elements at the source. The vertical resolution is the highest
    allowable given the time step.
    """

    dr = max(get_min_dr(**_OVERRIDES), 50)
    with config.override(dr_init=dr, **_OVERRIDES):
        ds = integrate()

        plot_integration(ds, f'plots/{config.name}/reference-integration.png')
        plot_ray_count(ds, f'plots/{config.name}/reference-ray-count.png')
        plot_boundary(ds, f'plots/{config.name}/reference-boundary.png')
        ds.to_netcdf(f'data/{config.name}/reference.nc')

def _get_descending_jets() -> xr.Dataset:
    """
    Generate a mean flow scenario with descending jets approximating the QBO.

    Returns
    -------
    xr.Dataset
        Dataset containing both components of the mean wind.

    """

    seconds = config.dt * np.arange(config.n_steps)
    time = cftime.num2date(seconds, f'seconds since {EPOCH}')
    z = InteractiveWind().z_centers

    k = 2 * np.pi / (3 * 86400)
    ell = 2 * np.pi / 25e3

    x, y = np.meshgrid(seconds, z)
    wave = np.exp(1j * (k * x + ell * y)).real.T
    env = np.exp(-((z - 45e3) / 10e3) ** 2)

    noise_1 = make_colored_noise(config.n_steps, config.n_grid - 1)
    noise_2 = make_colored_noise(config.n_steps, config.n_grid - 1)

    u = 30 * env * (wave + noise_1) + 5 * noise_2
    u[:, 1:-1] = shapiro_filter(u.T).T
    v = np.zeros_like(u)

    data = {'time' : time, 'z_centers' : z}
    data['u'] = (('time', 'z_centers'), u)
    data['v'] = (('time', 'z_centers'), v)

    return xr.Dataset(data)
