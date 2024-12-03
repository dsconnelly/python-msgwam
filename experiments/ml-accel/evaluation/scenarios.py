import cftime
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from msgwam import config
from msgwam.constants import EPOCH
from msgwam.plotting import plot_time_series
from msgwam.sources.spectra import _gaussians
from msgwam.utils import get_vertical_grids, make_colored_noise, shapiro_filter

from .strategies import get_overrides

_PERIOD_BOUNDS = [1, 7]
_WVL_BOUNDS = [17, 23]

def save_descending_jets() -> None:
    """
    Save a mean wind field consisting of descending jets alternating westerly
    and easterly. The mean state is saved at the reference resolution, from
    which it can be coarsened later.
    """

    with config.override(n_day=30, **get_overrides('reference')):
        ds = _get_descending_jets()
    
    _, cbar = plot_time_series(ds['u'], 50, cmap='PuOr_r')
    cbar.set_label('$\\bar{u}$ (m / s)')
    plt.tight_layout()

    ds.to_netcdf(f'data/{config.name}/descending-jets.nc')
    plt.savefig(f'plots/{config.name}/descending-jets.png', dpi=400)

def save_spectrum() -> None:
    """
    Save source spectrum data to disk, so that time is saved at the start of
    each integration and random perturbations are repeatable.
    """

    with config.override(spectrum_type='gaussians', dt=30, n_source=int(1e3)):
        _gaussians().to_netcdf(config.spectrum_file)

def _get_descending_jets() -> xr.Dataset:
    """
    Generate a mean wind time series with descending zonal jets and zero
    meridional wind.

    Returns
    -------
    xr.Dataset
        Dataset containing the mean wind time series.

    """

    seconds = config.dt * np.arange(config.n_steps)
    time = cftime.num2date(seconds, f'seconds since {EPOCH}')
    _, z = get_vertical_grids()

    rng = np.random.default_rng(7278)
    args = [seconds, 15 * 86400, 5 * 86400]
    period = make_colored_noise(*args, *_PERIOD_BOUNDS, rng) * 86400
    wvl = make_colored_noise(*args, *_WVL_BOUNDS, rng) * 1e3
    
    _, y = np.meshgrid(seconds, z)
    k = np.cumsum(1 / period) * config.dt
    ell = (1 / wvl) * y
    
    env_1 = np.exp(-((z - 45e3) / 10e3) ** 2)
    env_2 = np.exp(-((z - 40e3) / 20e3) ** 2)

    cutoffs = [30 * 60, 0]
    noise = make_colored_noise([seconds, z], [period, 15e3], cutoffs)
    wave = np.exp(1j * 2 * np.pi * (k + ell)).real.T

    u = 40 * env_1 * wave + 15 * env_2 * noise
    u[:, 1:-1] = shapiro_filter(u.T).T
    v = np.zeros_like(u)

    data = {'time' : time, 'z_centers' : z}
    data['u'] = (('time', 'z_centers'), u)
    data['v'] = (('time', 'z_centers'), v)

    return xr.Dataset(data)