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
    'maldives' : 4,

    'buenos-aires' : 7,
    'perth' : 7,

    'amundsen-sea' : 7,
    'weddell-sea' : 7
}

_N_MIN = 2 * np.pi / (2 * 3600)

def get_mima_scenario() -> xr.Dataset:
    """Generate a mean wind from MiMA outputs."""

    _, z = get_vertical_grids()
    kwargs = {'fill_value' : 'extrapolate'}
    data = {}

    with xr.open_dataset('data/mima-scenarios.nc') as ds:
        name = '-'.join(config.name.split('-')[1:])
        keep = ds['time.month'] == _MONTHS[name]
        ds = ds.sel(site=name).isel(time=keep)

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
