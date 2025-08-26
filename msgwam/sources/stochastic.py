import numpy as np

from .. import config

from .base import Source

class StochasticSource(Source):
    def _postprocess(
        self, *,
        cg_r: np.ndarray,
        data: np.ndarray,
        cdx: np.ndarray,
        **_
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return an appropriately reduced and randomly sampled set of ray volumes,
        such that the average wait time between launches is 1 / config.epsilon
        what it would be in the absence of randomness.
        """

        rate = config.epsilon * cg_r * config.dt / config.dr_source
        idx = np.repeat(np.arange(len(cdx)), np.random.poisson(rate))

        return data[:, idx], cdx[idx]
