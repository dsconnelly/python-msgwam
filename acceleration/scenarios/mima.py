import cftime
import numpy as np
import xarray as xr

from msgwam import config
from msgwam.constants import EPOCH
from msgwam.utils import get_vertical_grids

_MONTHS = {
    'copenhagen' : 1,
    'anchorage' : 1,

    'new-york' : 1,
    'lisbon' : 1,

    'miami' : 4,
    'brisbane' : 10,

    'singapore' : 10,
    'jakarta' : 4,

    'santiago' : 7,
    'buenos-aires' : 7,

    'amundsen-sea' : 7,
    'weddell-sea' : 7
}

_N_MIN = 2 * np.pi / (2 * 3600)

def get_mima_scenario() -> xr.Dataset:
    """Generate a mean wind from MiMA outputs."""

    data = {}
    with xr.open_dataset('data/mima-scenarios.nc') as ds:
        name = '-'.join(config.name.split('-')[1:])
        keep = ds['time.month'] == _MONTHS[name]
        ds = ds.sel(site=name).isel(time=keep)

        z = ds['z'].mean('time').values
        time = cftime.date2num(ds['time'].values, f'minutes since {EPOCH}')
        time = cftime.num2date(time - time[0], f'minutes since {EPOCH}')
        data = {'time' : time, 'z_centers' : z}

        data['u'] = (('time', 'z_centers'), ds['u'].values)
        data['v'] = (('time', 'z_centers'), ds['v'].values)

        data['rho'] = (('time', 'z_centers'), ds['rho'].values)
        data['N2'] = (('time', 'z_centers'), ds['N2'].values)
        data['G2'] = (('time', 'z_centers'), ds['G2'].values)

        data['flux_x'] = (('time', 'z_centers'), ds['gw_flux_x'].values)
        data['flux_y'] = (('time', 'z_centers'), ds['gw_flux_y'].values)

    _, z_centers = get_vertical_grids()
    kwargs = {'fill_value' : 'extrapolate'}
    ds = xr.Dataset(data).interp(z_centers=z_centers, kwargs=kwargs)
    ds['N2'] = np.maximum(ds['N2'], _N_MIN ** 2)

    return ds
