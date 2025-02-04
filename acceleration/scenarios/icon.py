import cftime
import numpy as np
import xarray as xr

from msgwam import config
from msgwam.constants import EPOCH
from msgwam.utils import get_time, get_vertical_grids

_REGIONS = {
    'midlatitudes' : (40.6, -74.0)
}

def get_ICON(data_dir: str) -> xr.Dataset:
    """
    Parse ICON output files and save a mean flow file for the tray tracer.

    Parameters
    ----------
    data_dir
        Directory where the ICON outputs are saved. Must contain files for `u`,
        `v`, and the vertical grid. The region of the globe will be determined
        based on `config.name`.

    """

    with xr.open_dataset(f'{data_dir}/vgrid.nc') as ds:
        z = ds['z_ifc'].isel(height=slice(None, -1))
        z = z.rename(ncells_2='ncells')

    with xr.open_dataset(f'{data_dir}/u.nc') as ds:
        lats = np.rad2deg(ds['clat'].values)
        lons = np.rad2deg(ds['clon'].values)
        u = ds['u']

    with xr.open_dataset(f'{data_dir}/v.nc') as ds:
        v = ds['v']

    lat, lon = _REGIONS[config.name.split('-')[-1]]
    keep = np.argsort((lat - lats) ** 2 + (lon - lons) ** 2)[:2]
    u, v, z = u.isel(ncells=keep), v.isel(ncells=keep), z.isel(ncells=keep)

    datetimes = u['time'].values
    seconds = (datetimes - datetimes[0]).astype(int) / 1e9
    time = cftime.num2date(seconds, f'seconds since {EPOCH}')

    data = {
        'time' : time,
        'level' : np.arange(z.shape[0]),
        'column' : np.arange(z.shape[1])
    }

    data['u'] = (('time', 'level', 'column'), u.values)
    data['v'] = (('time', 'level', 'column'), v.values)
    ds = xr.Dataset(data, coords={'z' : (('level', 'column'), z.values)})

    with config.override(n_grid=61):
        bins, labels = get_vertical_grids()

    g = ds.groupby_bins('z', bins, labels=labels)
    ds = g.mean('stacked_level_column').rename(z_bins='z_centers')

    time = get_time()
    _, z_centers = get_vertical_grids()
    kwargs = {'fill_value' : 'extrapolate'}
    ds = ds.interp(time=time, z_centers=z_centers, kwargs=kwargs)

    return ds.transpose()
