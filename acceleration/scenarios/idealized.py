from typing import Optional

import cftime
import numpy as np
import xarray as xr

from msgwam import config
from msgwam.constants import EPOCH
from msgwam.utils import get_vertical_grids, make_colored_noise

from ..hyperparameters import scenarios as hp

from .utils import get_background_noise

def get_descending_jets(seed: int=256) -> xr.Dataset:
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

    scales = [hp.time_scale_decay * 86400, hp.time_scale_cutoff * 86400]
    bounds = [hp.osc_period_min * 86400, hp.osc_period_max * 86400]
    period = make_colored_noise(seconds, *scales, *bounds, rng)
    k = np.cumsum(1 / period)[:, None] * config.dt

    t = (z - hp.osc_bottom) / (hp.osc_top - hp.osc_bottom)
    amp = (1 - t) * hp.osc_amp_min + t * (hp.osc_amp_max)
    amp = np.clip(amp, hp.osc_amp_min, hp.osc_amp_max)

    wvl = (1 - t) * hp.osc_wvl_bottom + t * hp.osc_wvl_top
    ell = np.cumsum(1 / wvl) * (z[1] - z[0])

    env = _make_env(z, hp.osc_bottom, hp.osc_top)
    u = env * amp * np.exp(2j * np.pi * (k + ell)).real    
    u = u + get_background_noise(seconds, z, rng)
    v = np.zeros_like(u)

    data = {'time' : time, 'z_centers' : z}
    data['u'] = (('time', 'z_centers'), u)
    data['v'] = (('time', 'z_centers'), v)

    return xr.Dataset(data)

def _make_env(
    z: np.ndarray,
    z_bot: Optional[float]=None,
    z_top: Optional[float]=None,
    decay: float=3e3
) -> np.ndarray:
    """
    Make an envelope that selects certain regions of the column.

    Parameters
    ----------
    z
        Array of vertical grid points.
    z_bot, z_top
        Lower and upper decay locations, respectively. If either is `None`, the
        envelope will not decay in that direction.
    decay
        Scale at which the rolloff should happen.

    Returns
    -------
    np.ndarray
        Array containing the value of the envelope at each point in `z`.

    """

    if isinstance(z_bot, np.ndarray) or isinstance(z_top, np.ndarray):
        env = np.ones((43201, 400))

    else:
        env = np.ones_like(z)

    if z_bot is not None:
        env[z < z_bot] = np.exp(-0.5 * ((z - z_bot) / decay) ** 2)[z < z_bot]

    if z_top is not None:
        env[z > z_top] = np.exp(-0.5 * ((z - z_top) / decay) ** 2)[z > z_top]

    return env
