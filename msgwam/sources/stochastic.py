from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from ..means import MeanState
from .. import config

from .base import Source

if TYPE_CHECKING:
    from ..means import MeanState

class StochasticSource(Source):
    def __init__(self):
        """
        The stochastic source keeps a record of whether or not it has been
        called, so that at the beginning of the run no random sampling is done.
        """

        super().__init__()
        self.called = False

    def _postprocess(
        self,
        _: MeanState,
        cg_r: np.ndarray,
        data: np.ndarray,
        cdx: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return an appropriately reduced and randomly sampled set of ray volumes,
        such that the average wait time between launches is 1 / config.epsilon
        what it would be in the absence of randomness.
        """

        if not self.called:
            self.called = True
            return data, cdx

        p = config.epsilon * cg_r * config.dt / config.dr_init
        keep = np.random.rand(data.shape[1]) < p

        return data[:, keep], cdx[keep]
