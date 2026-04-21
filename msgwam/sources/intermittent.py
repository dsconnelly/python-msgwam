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
        self._timer = config.dr_source * np.random.rand(config.n_source)
        self._timer = self._timer - np.sqrt(config.epsilon) * config.dr_source

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

        idx = self._timer <= 0
        distances = config.epsilon * cg_r * config.dt
        self._timer[~idx] = self._timer[~idx] - distances[~idx]
        self._timer[idx] = config.dr_source

        return data[:, idx], cdx[idx]
