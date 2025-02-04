from typing import Optional

import cftime
import numpy as np
import xarray as xr

from msgwam import config
from msgwam.constants import EPOCH
from msgwam.utils import get_vertical_grids, make_colored_noise

from ..hyperparameters import scenarios as hp

def get_descending_jets(seed: int=6909086) -> xr.Dataset:
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

    osc_top = config.z_max - 10e3
    osc_bot = osc_top - hp.osc_wavelength
    ell = 1 / hp.osc_wavelength

    center = make_colored_noise(seconds, *scales, -1, 1, rng)[:, None]
    center = hp.osc_center_max * np.sign(center) * (abs(center) ** 1.3)
    width = 60 - 40 * abs(center) / hp.osc_center_max

    env = _make_env(z, osc_bot, osc_top)
    wave = np.exp(2j * np.pi * (k + ell * z)).real
    u = env * (center + width * wave)

    jet = np.exp(2j * np.pi * seconds / hp.lower_period / 86400).real[:, None]
    u = u + hp.lower_amplitude * _make_env(z, z_top=osc_bot) * jet

    decays, cutoffs = [3 * 86400, 5e3], [2 * 86400, 3e3]
    noise = make_colored_noise([seconds, z], decays, cutoffs, -1, 1, rng)
    u = u + hp.noise_amplitude * noise

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

    env = np.ones_like(z)

    if z_bot is not None:
        env[z < z_bot] = np.exp(-0.5 * ((z[z < z_bot] - z_bot) / decay) ** 2)

    if z_top is not None:
        env[z > z_top] = np.exp(-0.5 * ((z[z > z_top] - z_top) / decay) ** 2)

    return env
