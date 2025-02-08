from typing import Optional

import numpy as np

from msgwam.utils import make_colored_noise

from ..hyperparameters import scenarios as hp

_DECAYS = [2 * 86400, 3000]
_CUTOFFS = [43200, 1000]

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