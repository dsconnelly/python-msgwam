from typing import Optional

import numpy as np

from msgwam.utils import make_colored_noise

from ..hyperparameters import scenarios as hp

_DECAYS = [2 * 86400, 5e3]
_CUTOFFS = [86400, 2e3]

def get_background_noise(
    seconds: np.ndarray,
    z: np.ndarray,
    rng: Optional[np.random.Generator]=None
) -> np.ndarray:
    """
    Make background noise. Extracted as a standalone function so that various
    mean flow functions can use the same logic.

    Parameters
    ----------
    seconds
        Array of times at which to generate noise.
    z
        Array of vertical levels at which to generate noise.
    rng
        Random state to pass to `make_colored_noise`.

    Returns
    -------
    np.ndarray
        Two-dimensional background noise array.

    """

    if rng is None:
        rng = np.random.default_rng()

    noise = make_colored_noise([seconds, z], _DECAYS, _CUTOFFS, -1, 1, rng)
    return hp.noise_amplitude * noise

def round_sigfigs(a: np.ndarray, n: int) -> np.ndarray:
    """
    Round data to a given number of significant figures. Due to Scott Gigante on
    StackExchange, adapted from https://stackoverflow.com/q/18915378.

    Parameters
    ----------
    a
        Data to round.
    n
        Number of significant figures to retain.

    Returns
    -------
    np.ndarray
        Rounded data.

    """

    a_pos = np.where(a != 0, abs(a), 10 ** (n - 1))
    mags = 10 ** (n - 1 - np.floor(np.log10(a_pos)))

    return np.round(a * mags) / mags