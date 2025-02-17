import cftime
import numpy as np
import xarray as xr

from msgwam import config
from msgwam.constants import EPOCH
from msgwam.utils import get_time, get_vertical_grids

from ..hyperparameters import scenarios as hp
from .utils import get_background_noise

_REGIONS = {
    'midlatitudes' : (41, -74),
    'vortex' : (60, 105)
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

    lat, lon = _REGIONS[hp.ICON_region]
    keep = np.argsort((lat - lats) ** 2 + (lon - lons) ** 2)[:hp.n_columns]
    u, v, z = u.isel(ncells=keep), v.isel(ncells=keep), z.isel(ncells=keep)
    z = z.values.mean(axis=1)

    datetimes = u['time'].values
    seconds = (datetimes - datetimes[0]).astype(int) / 1e9
    time = cftime.num2date(seconds, f'seconds since {EPOCH}')
    data = {'time' : time, 'z' : z, 'column' : np.arange(hp.n_columns)}

    data['u'] = (('time', 'z', 'column'), u.values)
    data['v'] = (('time', 'z', 'column'), v.values)
    ds = xr.Dataset(data).mean('column')

    time = get_time()
    _, z_centers = get_vertical_grids()
    kwargs = {'fill_value' : 'extrapolate'}
    ds = ds.interp(time=time, z=z_centers, kwargs=kwargs)

    rng = np.random.default_rng(123)
    seconds = cftime.date2num(time, f'seconds since {EPOCH}')
    ds['u'] = ds['u'] + get_background_noise(seconds, z_centers, rng)
    ds['v'] = ds['v'] + get_background_noise(seconds, z_centers, rng)

    return ds.rename(z='z_centers')
