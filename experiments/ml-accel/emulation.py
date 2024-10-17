import cftime
import matplotlib.pyplot as plt
import os
import torch
import xarray as xr

from tqdm import trange

from msgwam import config
from msgwam.constants import EPOCH
from msgwam.dispersion import cg_r
from msgwam.mean import MeanState
from msgwam.plotting import plot_time_series
from msgwam.sources import SimpleSource
from msgwam.utils import open_dataset

from utils import DATA_DIR, dimensionalize

def emulate_fluxes() -> None:
    """
    Use the neural network to predict momentum fluxes with prescribed mean wind,
    and save the result as a netCDF file.
    """

    # tag = 'best-r0'
    # path = f'{DATA_DIR}/surrogate-fine/model-{tag}.jit'
    path = 'data/ml-accel/surrogate-fine/model-best-r0.jit'
    model = torch.jit.load(path)

    config.n_source //= 3
    config.dr_init *= 3
    config.refresh()

    Y = SimpleSource().data[:, 0].T
    _, dr, k, l, m, dk, dl, dm, dens = Y.T
    config.reset()

    z = MeanState().z_centers
    seconds = config.dt * torch.arange(config.n_t_max)
    time = cftime.num2date(seconds, f'seconds since {EPOCH}')

    prefix = DATA_DIR + '/../../'
    with open_dataset(prefix + config.prescribed_wind) as ds:
        ds = ds.interp(time=time)
        u = torch.as_tensor(ds['u'].values)
        u = u[:, None].expand(-1, Y.shape[0], -1)

    cg_source = cg_r(k, l, m)
    T = 1 * (z.max() - z.min()) / cg_source
    T = torch.minimum(T, torch.as_tensor(86400 * 2))

    n_persist = torch.round(T / config.dt).int()
    max_persist = int(n_persist.max())

    delay = 8e3 / cg_source
    n_delay = torch.round(delay / config.dt).int()
    n_delay = torch.minimum(n_delay, n_persist)

    weights = torch.zeros((max_persist, Y.shape[0], 1))
    for j, n in enumerate(n_persist):
        weights[n_delay[j]:n, j] = 1

    n_skips = torch.ceil(dr / (cg_source * config.dt))
    output = torch.zeros((config.n_t_max, config.n_grid))

    with torch.no_grad():
        for i in trange(config.n_t_max):
            Z = dimensionalize(Y, model(u[i], Y), T - delay)
            Z[torch.remainder(i, n_skips) != 0] = 0

            n = min(max_persist, config.n_t_max - i)
            profile = (weights[:n] * Z).sum(dim=1)
            output[i:(i + n)] += profile

    ds = xr.Dataset({
        'z' : z,
        'time' : time,
        'u' : (('time', 'z'), u[:, 0]),
        'pmf_u' : (('time', 'z'), (output[:, :-1] + output[:, 1:]) / 2)
    })
    
    plot_time_series(1000 * ds['pmf_u'], 2)
    plt.tight_layout()

    ds.to_netcdf(f'{DATA_DIR}/../{config.name}/few-emulation-prescribed.nc')
    path = f'{DATA_DIR}/../../plots/{config.name}/emulated-fluxes.png'
    plt.savefig(os.path.abspath(path), dpi=400)
