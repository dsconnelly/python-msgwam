import numpy as np

from .. import config

from .base import Source

class IntermittentSource(Source):
    def __init__(self) -> None:
        """
        The intermittent source needs to initialize an array to hold the timers
        until the next launch of each source channel.
        """

        super().__init__()
        self._timer = np.zeros(config.n_source, dtype=np.int_)

    def _postprocess(
        self, *,
        cg_r: np.ndarray,
        data: np.ndarray,
        cdx: np.ndarray,
        **_
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Each time a ray volume is launched, a countdown timer is set until the
        next launch based on its current speed. This function launches the rays
        whose timer has reached zero and resets their timers, and decrements the
        timers for the remaining rays.
        """

        ns = config.dr_source / (config.epsilon * config.dt * cg_r)
        round_up = np.random.rand(config.n_source) < ns - np.floor(ns)
        ns = np.floor(ns) + round_up.astype(int)

        idx = self._timer == 0
        self._timer[idx] = ns[idx]
        self._timer[~idx] = self._timer[~idx] - 1
        
        return data[:, idx], cdx[idx]