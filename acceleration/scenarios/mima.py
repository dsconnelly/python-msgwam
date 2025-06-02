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

        grav, c_p, N_min = 9.8, 10004, 0.005
        T = xr.DataArray(ds['T'].values, {'T' : time, 'z' : z})
        N2 = ((grav / T) * (T.differentiate('z') + grav / c_p))
        N2 = np.maximum(N2, N_min ** 2)

        data['u'] = (('time', 'z_centers'), ds['u'].values)
        data['v'] = (('time', 'z_centers'), ds['v'].values)
        data['N'] = (('time', 'z_centers'), np.sqrt(N2.values))

        data['flux_x'] = (('time', 'z_centers'), ds['gw_flux_x'].values)
        data['flux_y'] = (('time', 'z_centers'), ds['gw_flux_y'].values)

    _, z_centers = get_vertical_grids()
    kwargs = {'fill_value' : 'extrapolate'}
    ds = xr.Dataset(data).interp(z_centers=z_centers, kwargs=kwargs)
    ds['N'] = np.maximum(ds['N'], N_min)

    return ds
