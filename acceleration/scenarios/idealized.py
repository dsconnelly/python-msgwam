from typing import Optional

import cftime
import numpy as np
import xarray as xr

from msgwam import config
from msgwam.constants import EPOCH
from msgwam.utils import get_vertical_grids, make_colored_noise as noise

from ..hyperparameters import scenarios as hp

def get_gated_oscillation(seed: int=177485) -> xr.Dataset:
    """
    Generate a mean wind scenario consisting of a low-level jet, which acts as a
    rapidly opening and closing gate to certain chunks of the source spectrum,
    and an upper-level descending oscillation, which provides shear zones in
    those waves that make it through the gate deposit momentum.
    """

    if seed is None:
        seed = np.random.randint(int(1e6))
        print(f'Using random seed {seed}')

    rng = np.random.default_rng(seed)
    n_steps = int(86400 * config.n_day / config.dt) + 1
    seconds = config.dt * np.arange(n_steps)
    days = seconds / 86400

    units = f'seconds since {EPOCH}'
    time = cftime.num2date(seconds, units)
    _, z = get_vertical_grids()

    a, b = hp.gate_open
    a, b = hp.gate_closed - a, a - b
    amp = a + b * noise(days, *hp.time_scales, 0, 1, rng) ** 2
    gate = hp.gate_closed - amp * np.cos(2 * np.pi * days / hp.gate_period) ** 4

    z_gate = noise(days, *hp.time_scales, *hp.z_gate_bounds, rng)
    env = np.exp(-0.5 * ((z - z_gate[:, None]) / hp.gate_width) ** 2)
    u = env * gate[:, None]

    tides = np.sin(2 * np.pi * (days[:, None] + z / hp.wvl))
    z_tide = noise(days, *hp.time_scales, *hp.z_tide_bounds, rng)
    env = _make_env(z, z_tide[:, None], width=hp.shear_width)
    u = u + env * hp.wave_amp * tides

    u = u + hp.noise_amp * noise(
        [days, z],
        hp.noise_decays,
        hp.noise_cutoffs,
        rng=rng
    )
    
    v = np.zeros_like(u)
    data = {'time' : time, 'z_centers' : z}
    data['u'] = (('time', 'z_centers'), u)
    data['v'] = (('time', 'z_centers'), v)

    return xr.Dataset(data)

def _make_env(
    z: np.ndarray,
    z_bot: Optional[float]=None,
    z_top: Optional[float]=None,
    width: float | tuple[float, float]=1e3
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
    width
        Scale at which the rolloff should happen.

    Returns
    -------
    np.ndarray
        Array containing the value of the envelope at each point in `z`.

    """

    if isinstance(z_bot, np.ndarray) or isinstance(z_top, np.ndarray):
        env = np.ones((config.n_steps, config.n_grid - 1))
    else:
        env = np.ones_like(z)

    if z_bot is not None:
        env[z < z_bot] = np.exp(-0.5 * ((z - z_bot) / width) ** 2)[z < z_bot]

    if z_top is not None:
        env[z > z_top] = np.exp(-0.5 * ((z - z_top) / width) ** 2)[z > z_top]

    return env
