from typing import Literal, Optional

import cftime
import numpy as np
import xarray as xr

from msgwam import config
from msgwam.constants import EPOCH
from msgwam.utils import get_vertical_grids, open_dataset

def get_rmse(a: xr.DataArray, b: xr.DataArray | Literal[0]=0) -> xr.DataArray:
    """
    Compute the root-mean-square error over time between two arrays. The second
    argument can also be passed in as zero, so that this function can be used to
    calculate the RMS value of a single data array.

    Parameters
    ----------
    a, b
        Data with which to compute RMS errors.

    Returns
    -------
    xr.DataArray
        Array of RMS errors, with the time dimension averaged out.

    """

    return np.sqrt(((a - b) ** 2).mean('time'))

def load_data(
    path: str,
    spinup_days: int=5,
    resample: Optional[str]='3h',
    var: str='flux'
) -> xr.DataArray:
    """
    Load data from an integration.

    Parameters
    ----------
    path
        Location of saved integration output, or the name of a defined strategy,
        in which case the actual path will be determined automatically.
    spinup_days
        Number of days to discard from the beginning of the integration.
    resample
        Resample string to apply to the flux data.
    var
        What data variable to return. Should be the name of a variable in the
        dataset or `'flux'`, in which case the total (westerly plus easterly)
        momentum flux time series is returned.

    Returns
    -------
    xr.DataArray
        Array of momentum fluxes, subselected and resampled as appropriate.
    
    """

    if not path.endswith('.nc'):
        path = f'data/{config.name}/{path}.nc'

    with open_dataset(path) as ds:
        z_faces, _ = get_vertical_grids()
        ds = ds.interp(z_faces=z_faces)

        if 'sample' in ds.coords:
            ds = ds.mean('sample')

        units = f'days since {EPOCH}'
        days = cftime.date2num(ds['time'], units)
        ds = ds.isel(time=(days >= spinup_days))

        if resample is not None:
            ds = ds.resample(time=resample).mean('time')

    if var == 'flux':
        return ds['pmf_e'] + ds['pmf_w']

    return ds[var]