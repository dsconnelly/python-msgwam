from typing import Any

import xarray as xr

from ... import hyperparameters as hp
from ...strategies import get_overrides as _get_strat_overrides

def get_overrides(n: int) -> dict[str, Any]:
    """
    Get the configurations to use while generating training data.

    Parameters
    ----------
    n
        Task ID for which to return configuration overrides.

    Returns
    -------
    dict[str, Any]
        Keyword arguments for `config.override`.

    """

    year, month, site, lat = get_info(n)
    path = f'data/ml-accel/context/{year}/{site}-{month}.nc'
    kwargs = _get_strat_overrides('eulerian', 'fine')

    kwargs['dt'] = hp.generation.dt
    kwargs['dt_output'] = hp.generation.dt_output
    kwargs['prescribed_mean_file'] = path
    kwargs['latitude'] = lat

    return kwargs

def get_info(n: int) -> tuple[int, int, str, float]:
    """
    Given an integer (should be the SLURM task ID) get information relevant to
    setting up or integrating a particular month of training data.

    Parameters
    ----------
    int
        Current task ID.

    Returns
    -------
    int, int
        Year and month to pull MiMA data from.
    str
        Site name.
    float
        Latitude of the site.

    """

    n_site, month = divmod(n, 24)
    year, month = divmod(month, 12)
    month = month + 1
    year = 25 - year

    with xr.open_dataset('data/mima-scenarios-25.nc') as ds:
        site = ds['site'].values[n_site]
        lat = ds['lat'].values[n_site]

    return year, month, site, lat
