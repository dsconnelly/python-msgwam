import xarray as xr

from msgwam import config
from msgwam.integration import integrate

from ..hyperparameters import strategies as hp

from .overrides import get_overrides

def get_integration(strategy: str) -> xr.Dataset:
    """
    Integrate the solver and return a dataset containing the outputs. Broken out
    as a separate function so that other code can perform integrations without
    saving results to disk.

    Parameters
    ----------
    strategy
        Name of the configuration strategy with which to integrate.

    Returns
    -------
    xr.Dataset
        Dataset containing the integration outputs. Will also contain an extra
        dimension `'member'` which either ranges over ensemble members, for
        configurations with some randomness, or is a singleton.

    """

    n = hp.n_ensemble if (config.jitter or strategy == 'stochastic') else 1
    datasets = [integrate().assign_coords(member=i) for i in range(n)]

    return xr.concat(datasets, dim='member')

def save_integration(strategy: str, *args: str) -> None:
    """
    Integrate with the configurations specific to the given strategy and save
    result to disk as a netCDF file. This is the function that is meant to be
    called from the command line.

    Parameters
    ----------
    strategy
        Name of the configuration strategy with which to integrate.
    args
        Arguments to pass to the override function, if any.

    """

    fname = '-'.join([strategy, *args])
    path = f'data/{config.name}/strategies/{fname}.nc'

    with config.override(**get_overrides(strategy, *args)):
        get_integration(strategy).to_netcdf(path)
