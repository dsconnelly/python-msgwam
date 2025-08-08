from typing import Literal, Optional

import cftime
import numpy as np
import xarray as xr

from msgwam import config
from msgwam.constants import EPOCH
from msgwam.utils import get_rho, get_vertical_grids, open_dataset

from ..shared.filtering import gaussian_filter

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
    field: str='flux_x',
    spinup_days: int=5,
    time_filter: Optional[int]=43200,
    z_filter: Optional[float]=4e3,
    ensemble_mean: bool=True
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
        variable in the dataset; `'flux_{x | y}'`, in which case the total flux
        in the specified direction is returned; or `'accleration_{x | y}'`, in
        which case the corresponding mean wind forcing is returned.
    spinup_days
        Number of days to discard from the beginning of the simulation.
    time_filter, z_filter
        Time and height scales at which to apply filters, in seconds and meters,
        respectively. A Gaussian filter with standard deviation equal to 1 / 4
        of this value will be applied, so that ~95% of the kernel mass is within
        `{time | z}_filter / 2` of the center.
    ensemble_mean
        Whether to take the ensemble mean, if there are multiple members.

    Returns
    -------
    xr.DataArray
        Array of requested data values, subselected and filtered as appropriate.

    """

    if z_filter is None and field.startswith('acceleration'):
        z_filter = 4e3

    if not path.endswith('.nc'):
        data_dir = f'data/{config.name}/strategies'
        path = f'{data_dir}/{path}.nc'

    with open_dataset(path) as ds:
        z_faces, z_centers = get_vertical_grids()
        ds = ds.interp(z_faces=z_faces, z_centers=z_centers)

        if field.startswith(('flux', 'acceleration')):
            a, b = {'x' : 'ew', 'y' : 'ns'}[field[-1]]
            data = ds[f'pmf_{a}'] + ds[f'pmf_{b}']

            if field.startswith('acceleration'):
                rho = xr.DataArray(get_rho(z_faces), [ds['z_faces']])
                data = -data.diff('z_faces') / (z_faces[1] - z_faces[0]) / rho
                data = data.rename(z_faces='z_centers')

        else:
            data = ds[field]

    if ensemble_mean and ('member' in data.coords):
        data = data.mean('member')

    units = f'days since {EPOCH}'    
    days = cftime.date2num(data['time'], units)
    data = data.isel(time=(days >= spinup_days))

    z_name = list(data.coords)[1]
    zipped = zip(['seconds', z_name], [time_filter, z_filter])
    kwargs = {k : v for k, v in zipped if v is not None}
    data = gaussian_filter(data, **kwargs)

    return data