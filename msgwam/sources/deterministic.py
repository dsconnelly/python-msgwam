import numpy as np

from .base import Source

class DeterministicSource(Source):
    def _launch(
        self,
        _, 
        n_step: int,
        cdx: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        A deterministic source returns the properties of all requested waves,
        along with the required second return value indicating as much.
        """

        return self._data[n_step][:, cdx], cdx
        