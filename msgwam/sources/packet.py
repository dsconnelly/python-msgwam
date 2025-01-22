from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from .. import config

from .base import Source

if TYPE_CHECKING:
    from ..means import MeanState

class PacketSource(Source):
    def _postprocess(
        self, *,
        data: np.ndarray,
        cdx: np.ndarray,
        **_
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        A packet source returns multiple copies of the ray properties in each
        slot. These are to be interpreted by the propagator as ray volumes
        queued up in vertical space.
        """

        data = np.repeat(data, config.n_repeat, axis=1)
        cdx = np.repeat(cdx, config.n_repeat)

        return data, cdx
