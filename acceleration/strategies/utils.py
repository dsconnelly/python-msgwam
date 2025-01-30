from typing import Literal, Optional

import cftime
import numpy as np
import xarray as xr

from scipy.ndimage import gaussian_filter1d as filter

from msgwam import config
from msgwam.constants import EPOCH
from msgwam.utils import get_rho, get_vertical_grids, open_dataset

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
    field: str='flux',
    spinup_days: int=5,
    filter_width: Optional[int]=21600
) -> xr.DataArray:
    """
    Load data from the integration of a particular strategy.

    Parameters
    ----------
    strategy
        Path from which to load integration output. Can also just be the name of
        a strategy, in which case the path will be chosen automatically.
    field
        Name of the data variable to return. Should be either the name of a
        variable in the dataset; `'flux'`, in which case the total (westerly
        plus easterly) momentum flux is returned; or `'acceleration'`, in which
        case the corresponding mean wind forcing is computed and returned.
    spinup_days
        Number of days to discard from the beginning of the simulation.
    filter_width
        Time scale at which to apply a filter, in seconds. A Gaussian filter
        with standard deviation equal to a 1 / 4 of this value will be applied,
        so that ~95% of the mass is within `filter_width / 2` of the center.

    Returns
    -------
    xr.DataArray
        Array of requested data values, subselected and filtered as appropriate.

    """

    if not path.endswith('.nc'):
        data_dir = f'data/{config.name}/strategies'
        path = f'{data_dir}/{path}.nc'

    with open_dataset(path) as ds:
        z_faces, z_centers = get_vertical_grids()
        ds = ds.interp(z_faces=z_faces, z_centers=z_centers)

        if field in ['flux', 'acceleration']:
            data = ds['pmf_e'] + ds['pmf_w']

            if field == 'acceleration':
                dz = z_faces[1] - z_faces[0]
                rho = xr.DataArray(get_rho(z_faces), [ds['z_faces']])
                data = -data.diff('z_faces') / dz / rho
                data = data.rename(z_faces='z_centers')

        else:
            data = ds[field]

    if 'member' in data.coords:
        data = data.mean('member')

    units = f'days since {EPOCH}'    
    days = cftime.date2num(data['time'], units)
    data = data.isel(time=(days >= spinup_days))

    if filter_width is not None:
        sigma = int(filter_width / (days[1] - days[0]) / 86400 / 4)
        filtered = filter(data.values, sigma, axis=0)
        data = xr.DataArray(filtered, data.coords)

    return data