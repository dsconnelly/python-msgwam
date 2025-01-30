import cftime
import numpy as np
import xarray as xr

from msgwam import config
from msgwam.constants import EPOCH
from msgwam.utils import get_vertical_grids, make_colored_noise

from ..hyperparameters import scenarios as hp

def get_descending_jets(seed: int) -> xr.Dataset:
    """
    Generate a mean wind scenario consisting of an upper-atmosphere oscillation
    with (possibly) varying period. The lower atmosphere features a much slower
    oscillation so that no section of the spectrum is systematically filtered.
    """

    rng = np.random.default_rng(seed)
    n_steps = int(86400 * config.n_day / config.dt) + 1
    seconds = config.dt * np.arange(n_steps)

    units = f'seconds since {EPOCH}'
    time = cftime.num2date(seconds, units)
    _, z = get_vertical_grids()
    dz = z[1] - z[0]
    
    scales = [5 * 86400, 5 * 86400]
    bounds = [hp.osc_period_min * 86400, hp.osc_period_max * 86400]
    period = make_colored_noise(seconds, *scales, *bounds, rng)
    wvl = hp.osc_wvl * np.ones_like(z)

    osc_bot = config.z_max - hp.osc_wvl
    env = np.exp(-0.5 * ((z - osc_bot) / hp.osc_width) ** 2)
    env[z > osc_bot] = 1

    k = np.cumsum(1 / period) * config.dt
    ell = np.cumsum(1 / wvl)[:, None] * dz
    wave = env * np.exp(2j * np.pi * (k + ell)).real.T

    z_meet = z[np.argmin(abs(env - hp.filter_amplitude / hp.osc_amplitude))]
    env = np.exp(-((z_meet - z) / (z_meet - config.z_min)) ** 2)
    k = seconds / hp.filter_period / 86400 + rng.random()
    filter = env * np.exp(2j * np.pi * k).real[:, None]

    u = hp.osc_amplitude * wave + hp.filter_amplitude * filter
    v = np.zeros_like(u)

    data = {'time' : time, 'z_centers' : z}
    data['u'] = (('time', 'z_centers'), u)
    data['v'] = (('time', 'z_centers'), v)

    return xr.Dataset(data)
