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
    'dt' : 30,
    'n_grid' : 201,
    'n_source' : 100,
    'n_max' : 500000,
    'n_increment' : 10000,
    'prune_by' : 'none'
}

def save_mean_state(*args) -> None:
    """
    Save the mean wind to be used for this configuration. The wind is saved at
    the reference resolution, from which it can be coarsened later.

    Parameters
    ----------
    args
        Arguments to pass to the function that generates the wind.

    """

    np.random.seed(1234)
    with config.override(**_OVERRIDES):
        ds = _get_descending_jets(*args)
        ds.to_netcdf(f'data/{config.name}/mean-state.nc')

def save_reference() -> None:
    """
    Save the reference integration to compute errors against. The reference run
    has no pruning and extremely high values for the time step and the number of
    spectral elements at the source. The vertical resolution is the highest
    allowable given the time step.
    """

    with config.override(**_OVERRIDES):
        dr = max(get_min_dr(), 50)
        print(f'Reference integration will take dr = {dr}')

    with config.override(dr_init=dr, **_OVERRIDES):
        ds = integrate()

        plot_integration(ds, f'plots/{config.name}/reference-integration.png')
        plot_ray_count(ds, f'plots/{config.name}/reference-ray-count.png')
        plot_boundary(ds, f'plots/{config.name}/reference-boundary.png')
        ds.to_netcdf(f'data/{config.name}/reference.nc')

def _get_descending_jets(period_days: str='2') -> xr.Dataset:
    """
    Generate a mean flow scenario with descending jets approximating the QBO.

    Parameters
    ----------
    period_days
        Period of the oscillation, in days. Because this parameter is passed in
        from the command line, we have to accept it as a string.

    Returns
    -------
    xr.Dataset
        Dataset containing both components of the mean wind.

    """

    seconds = config.dt * np.arange(config.n_steps)
    time = cftime.num2date(seconds, f'seconds since {EPOCH}')
    z = InteractiveWind().z_centers
    cutoff_scales = [30 * 60, 0]

    period = float(period_days) * 86400
    k, ell = 2 * np.pi / period, 2 * np.pi / 25e3
    x, y = np.meshgrid(seconds, z)
    
    env_1 = np.exp(-((z - 45e3) / 10e3) ** 2)
    env_2 = np.exp(-((z - 40e3) / 20e3) ** 2)
    wave = np.exp(1j * (k * x + ell * y)).real.T

    noise_1 = make_colored_noise([seconds, z], [period, 15e3], cutoff_scales)
    noise_2 = make_colored_noise([seconds, z], [9 * 3600, 5e3], cutoff_scales)

    u = env_1 * (60 * wave + 5 * noise_1) + env_2 * 10 * noise_2
    u[:, 1:-1] = shapiro_filter(u.T).T
    v = np.zeros_like(u)

    data = {'time' : time, 'z_centers' : z}
    data['u'] = (('time', 'z_centers'), u)
    data['v'] = (('time', 'z_centers'), v)

    return xr.Dataset(data)
