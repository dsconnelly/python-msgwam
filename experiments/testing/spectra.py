import numpy as np
import torch
import xarray as xr

from msgwam import config
from msgwam.constants import PROP_NAMES
from msgwam.dispersion import cg_r, m_from
from msgwam.spectra import _gaussians
from msgwam.utils import open_dataset, shapiro_filter

from scenarios import _REGIMES

CONV_WIDTH = 15
EPSILON = 0.05
DELTA_H = 5e3
DELTA_T = 20 * 60

def save_ICON_spectra() -> None:
    """
    Parse ICON output files and save a time-varying source spectrum.
    """

    config.n_source *= 5
    config.refresh()

    with _gaussians().sortby('cp_x') as ds:
        cs = torch.as_tensor(ds['cp_x'].values)
        bg = torch.as_tensor(ds.to_array().values)[:, None]

        k, l, m, dk, dl, dm, dens = bg
        bg_flux = abs(k * (dens * dk * dl * dm) * cg_r(k, l, m))

    regime = config.name.split('-')[-1]
    with xr.open_dataset(f'data/ICON/DJF-2324.nc') as ds:
        lats = np.rad2deg(ds['clat'].values)
        lons = np.rad2deg(ds['clon'].values)
        lat, lon = _REGIMES[regime]

        i = np.argmin((lats - lat) ** 2 + (lons - lon) ** 2)
        amplitude = torch.as_tensor(ds['ctmfl_cgw_a'].isel(ncells=i).values)

    with open_dataset(f'data/{config.name}/mean-state-ICON.nc') as ds:
        kwargs = {'z' : config.r_launch, 'method' : 'nearest'}
        u_source = torch.as_tensor(ds['u'].sel(**kwargs).values)
        time = ds['time'].values

    arg = cs - u_source[:, None]
    env = torch.exp(-0.5 * (arg / CONV_WIDTH) ** 2)
    env = env / env.sum(dim=1)[:, None]

    conv_flux = EPSILON * amplitude[:, None] * env
    spectrum = bg.expand(-1, conv_flux.shape[0], -1).clone()

    k = 2 * ((1 + arg ** 2 * (DELTA_T / DELTA_H) ** 2) ** -0.5) / DELTA_H
    k[conv_flux == 0] = 2 * torch.pi / config.wvl_hor_char
    k = k * torch.sign(cs).double()

    k[1:-1] = shapiro_filter(k)
    m = m_from(k, 0, cs)

    dc = cs[1] - cs[0]
    dk, dl = spectrum[3:5]
    dm = abs(k * dc / cg_r(k, 0, m))
    dens = (conv_flux + bg_flux) / (k ** 2 * dk * dl * dc)

    spectrum[0] = k
    spectrum[2] = m
    spectrum[5] = dm
    spectrum[6] = dens

    data = {'time' : time, 'cp_x' : cs}
    for i, prop in enumerate(PROP_NAMES[2:-2]):
        data[prop] = (('time', 'cp_x'), spectrum[i])

    path = f'data/{config.name}/ICON-spectra.nc'
    xr.Dataset(data).to_netcdf(path)
    config.reset()
