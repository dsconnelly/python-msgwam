import sys

import cftime
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

sys.path.insert(0, '.')
from msgwam import config
from msgwam.constants import EPOCH
from msgwam.plotting import plot_time_series

def save_mean_state(scenario: str) -> None:
    """
    Save a mean state file for this config setup.

    Parameters
    ----------
    scenario
        Name of the background scenario. Must correspond to the name of one of
        the functions in scenarios.py.

    """

    func_name = '_' + scenario.replace('-', '_')
    ds: xr.Dataset = globals()[func_name]()
    
    _, cbar = plot_time_series(ds['u'], 12)
    cbar.set_label('$\\bar{u}$ (m s$^{-1}$)')

    plt.tight_layout()
    plt.savefig(f'plots/{config.name}/mean-state.png')
    ds.to_netcdf(f'data/{config.name}/mean-state.nc')

def _oscillating_jets() -> xr.Dataset:
    """
    Return a mean flow scenario with two oscillating jets.

    Returns
    -------
    xr.Dataset
        Dataset containing the mean flow scenario.

    """

    faces = np.linspace(*config.grid_bounds, config.n_grid)
    centers = (faces[:-1] + faces[1:]) / 2

    width = 3e3
    low = 10 * np.exp(-((centers - 25e3) / width) ** 2)
    high = 10 * np.exp(-((centers - 38e3) / width) ** 2)

    n_days = 35
    seconds = 86400 * np.linspace(0, n_days, 24 * n_days)
    datetimes = cftime.num2date(seconds, f'seconds since {EPOCH}')
    envelope = np.sin(2 * np.pi * seconds / 86400 / 30)

    shape = (len(seconds), len(centers))
    dipole = np.broadcast_to(low - high, shape)
    u = envelope[:, None] * dipole
    v = np.zeros_like(u)

    return xr.Dataset({
        'z' : centers,
        'time' : datetimes,
        'u' : (('time', 'z'), u),
        'v' : (('time', 'z'), v)
    })
