import numpy as np

from .base import Source

class ConstantSource(Source):
    def _postprocess(
        self, *,
        data: np.ndarray,
        cdx: np.ndarray,
        **_
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        A constant-flux source returns the properties of all requested waves
        unchanged, along with the required second array indicating as much.
        """

        return data, cdx
