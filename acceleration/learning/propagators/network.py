from __future__ import annotations
from typing import TYPE_CHECKING, Self

import numpy as np
import torch

from msgwam import config
from .eulerian import EulerianPropagator


if TYPE_CHECKING:
    from msgwam.means import MeanState

class NetworkPropagator(EulerianPropagator):
    def __init__(self, mean: MeanState):
        """
        The NetworkPropagator must load the trained network before calling the
        parent initialization, so that it is available for the first step.
        """

        edges_coarse = self._allocate_bins(6, 0.9)
        edges_fine = self._allocate_bins(config.n_c, 0.9)
        cpt = (edges_fine[:-1] + edges_fine[1:]) / 2

        self._jdx = np.digitize(cpt, edges_coarse) - 1
        self._model = torch.jit.load(config.model_path)
        self._cpt_coarse = ((edges_coarse[:-1] + edges_coarse[1:]) / 2)[:, None]

        super().__init__(mean)

    def _get_wvn(self, mean: MeanState) -> np.ndarray:
        """
        
        """

        wind = np.vstack((mean.u, mean.v, -mean.u, -mean.v))
        N = np.vstack((mean.N, mean.N, mean.N, mean.N))
        f = abs(config.f) * np.ones((4, 1))

        C = np.hstack((wind, N, f))
        # M = self._M.reshape(4, -1, self._M.shape[-1])

        M = np.zeros((4, 6, config.n_grid - 1))
        for q in range(4):
            np.add.at(M[q], self._jdx, self._M[q, 0])

        with torch.no_grad():
            inputs = map(torch.as_tensor, [C, M])
            T_hat = self._model(*inputs).numpy()

        omega_hat = 2 * torch.pi / T_hat[:, None]
        wvn = np.maximum(omega_hat - f[:, None, None], 1e-8) / self._cpt_coarse

        out = np.zeros_like(self._M)

        for q in range(4):
            out[q, 0] = wvn[q, 0, self._jdx]

        # wvl = 1 / self._edges_wvn
        # wvn = 1 / ((wvl[0] + wvl[-1]) / 2)

        return out

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
