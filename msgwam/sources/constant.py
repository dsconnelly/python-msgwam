from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from .base import Source

if TYPE_CHECKING:
    from ..means import MeanState

class ConstantSource(Source):
    def _postprocess(
        self,
        n_step: int,
        mean: MeanState,
        cg_r: np.ndarray,
        data: np.ndarray,
        cdx: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        A constant-flux source returns the properties of all requested waves
        unchanged, along with the required second array indicating as much.
        """

        return data, cdx
