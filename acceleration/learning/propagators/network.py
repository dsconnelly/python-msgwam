from __future__ import annotations
from typing import TYPE_CHECKING, Self

import numpy as np
import torch

from msgwam import config
from msgwam.dispersion import get_omega_hat

from .eulerian import EulerianPropagator
from .utils import get_bin_edges

if TYPE_CHECKING:
    from msgwam.means import MeanState

class NetworkPropagator(EulerianPropagator):
    def __init__(self, mean: MeanState):
        """
        The NetworkPropagator must load the trained network before calling the
        parent initialization, so that it is available for the first step.
        """

        self._model = torch.jit.load(config.model_path)
        super().__init__(mean)

    def _get_wvn(self, mean: MeanState) -> np.ndarray:
        """
        
        """

        wind = np.vstack((mean.u, mean.v, -mean.u, -mean.v))
        N = np.vstack((mean.N, mean.N, mean.N, mean.N))
        f = abs(config.f) * np.ones((4, 1))

        C = np.hstack((wind, N, f))
        M = self._M.reshape(4, -1, self._M.shape[-1])

        with torch.no_grad():
            inputs = map(torch.as_tensor, [C, M])
            T_hat = self._model(*inputs).numpy()

        omega_hat = 2 * torch.pi / T_hat[:, None]
        return np.maximum(omega_hat - f[:, None, None], 1e-8) / self._cpt

import numba as nb
@nb.njit
def apply_smoothing(a: np.ndarray) -> np.ndarray:
    """
    Apply a Shapiro filter along the last dimension, while respecting initial
    zeros and so not polluting levels below the source.

    Parameters
    ----------
    a
        Array to filter.

    Returns
    -------
    a
        Array filtered along the last dimension.

    """

    out = np.zeros_like(a)
    for idx in np.ndindex(a.shape[:-1]):
        start = np.argmax(a[idx] != 0)

        for k in range(start, a.shape[-1]):
            out[*idx, k] += a[*idx, max(k - 1, start)]
            out[*idx, k] += a[*idx, min(k + 1, a.shape[-1] - 1)]
            out[*idx, k] += 2 * a[*idx, k]

    return out / 4
