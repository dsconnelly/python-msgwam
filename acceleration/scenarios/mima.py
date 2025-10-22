import cftime
import numpy as np
import xarray as xr

from typing import Optional

from msgwam import config
from msgwam.constants import EPOCH
from msgwam.utils import get_vertical_grids

from ..shared.constants import MIMA_MONTHS

_N_MIN = 2 * np.pi / (2 * 3600)

def get_mima_scenario(year: int=25, month: Optional[int]=None) -> xr.Dataset:
    """
    Generate a mean wind from MiMA outputs.
    
    Parameters
    ----------
    year
        Year to pull data from, indicated in the filename of the MiMA output.
    month
        Month within that year (1-12). If not provided, uses the month from the
        shared dictionary `MIMA_MONTHS`.

    Returns
    -------
    xr.Dataset
        Dataset containing the data needed to drive MS-GWaM.

    """

    _, z = get_vertical_grids()
    kwargs = {'fill_value' : 'extrapolate'}
    data = {}

    with xr.open_dataset(f'data/mima-scenarios-{year}.nc') as ds:
        name = '-'.join(config.name.split('-')[1:])
        ds = ds.sel(site=name)

        if month is None:
            month = MIMA_MONTHS[name]

        keep = ds['time.month'] == month
        ds = ds.isel(time=keep)

        time = cftime.date2num(ds['time'].values, f'minutes since {EPOCH}')
        time = cftime.num2date(time - time[0], f'minutes since {EPOCH}')
        data = {'time' : time, 'z_centers' : z}

        shape = (len(ds['time']), len(z))
        for vname in ['u', 'v', 'rho', 'N2', 'G2', 'flux_x', 'flux_y']:
           data[vname] = (('time', 'z_centers'), np.zeros(shape))

        for i in range(shape[0]):
            dsi = ds.isel(time=i)
            dsi = dsi.assign_coords(pfull=dsi['z'].values)
            dsi = dsi.interp(pfull=z, kwargs=kwargs)

            for vname in data:
                if vname in ['time', 'z_centers']:
                    continue

                dname = ('gw_' if vname.startswith('flux') else '') + vname
                data[vname][1][i] = dsi[dname].values

    ds = xr.Dataset(data)
    ds['N2'] = np.maximum(ds['N2'], _N_MIN ** 2)

    return ds
