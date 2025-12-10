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

    def _get_cg_r(self, mean: MeanState) -> np.ndarray:
        """The group velocity is obtained by calling the network."""

        wind = np.vstack((mean.u, mean.v, -mean.u, -mean.v))
        N = np.vstack((mean.N, mean.N, mean.N, mean.N))
        f = abs(config.f) * np.ones((4, 1))

        C = np.hstack((wind, N, f))
        M = self._M.reshape(4, -1, self._M.shape[-1])

        with torch.no_grad():
            inputs = map(torch.as_tensor, [C, M])
            cg = self._model(*inputs)

        return cg.numpy().reshape(4, *self._M.shape[1:])