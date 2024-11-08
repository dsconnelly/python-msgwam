from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from msgwam.means import MeanState

from .. import config
from ..dispersion import get_cg_r, get_m

from .base import Source

if TYPE_CHECKING:
    from ..means import MeanState

class StochasticSource(Source):
    def __init__(self) -> None:
        """
        At initialization, the stochastic source estimates the launch rate as a
        function of time. The group velocities calculated here are not saved,
        because they do not account for the changing mean state conditions.
        """

        super().__init__()

        k, l, *_ = self._data.transpose(1, 0, 2)
        m = get_m(k, l, self._cp_x, config.N_ref)
        cg_r = get_cg_r(k, l, m, config.N_ref)

        steps_per_launch = 0.5 * config.dr_init / (cg_r * config.dt)
        rates = (1 / np.ceil(steps_per_launch)).sum(axis=1)
        self.launch_rate = config.epsilon * rates.mean()

    def _postprocess(
        self,
        mean: MeanState,
        cg_r: np.ndarray,
        data: np.ndarray,
        cdx: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return an appropriately reduced and randomly sampled set of ray volumes,
        selected such that faster ray volumes are launched more frequently.
        """

        size = int(np.floor(self.launch_rate))
        if np.random.rand() < self.launch_rate - size:
            size = size + 1

        keep = np.random.choice(
            len(cdx),
            size=size,
            replace=False,
            p=(cg_r / cg_r.sum())
        )

        return data[:, keep], cdx[keep]
