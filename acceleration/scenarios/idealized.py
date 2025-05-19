from typing import Optional

import cftime
import numpy as np
import xarray as xr

from msgwam import config
from msgwam.constants import EPOCH
from msgwam.utils import get_vertical_grids, make_colored_noise

from ..hyperparameters import scenarios as hp

from .utils import get_background_noise

def get_descending_jets(seed: int=123) -> xr.Dataset:
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

    decays = [86400 * hp.time_scale_decay, hp.height_scale_decay]
    cutoffs = [86400 * hp.time_scale_cutoff, hp.height_scale_cutoff]
    bounds = np.array([hp.wvl_min, hp.wvl_max]) ** (1 / hp.wvl_power)

    wvl = make_colored_noise([seconds, z], decays, cutoffs, *bounds, rng)
    phase = np.cumsum(1 / wvl ** hp.wvl_power, axis=1) * (z[1] - z[0])

    bounds = [86400 * hp.period_min, 86400 * hp.period_max]
    period = make_colored_noise(seconds, decays[0], cutoffs[0], *bounds, rng)
    k = (np.diff(phase, axis=0, append=phase[-1:]) / config.dt).mean(axis=1)
    phase = phase + np.cumsum(1 / period - k)[:, None] * config.dt

    t = (z - hp.osc_bottom) / (hp.osc_top - hp.osc_bottom)
    amp = (1 - t) * hp.noise_amplitude + t * (hp.amp_max)
    amp = np.clip(amp, hp.noise_amplitude, hp.amp_max)

    env = _make_env(z, hp.osc_bottom, hp.osc_top)
    u = env * amp * np.exp(2j * np.pi * phase).real    
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
    decay: float=1e3
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
