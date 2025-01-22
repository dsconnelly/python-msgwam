from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
import torch

from .. import config
from ..utils import get_wavenumbers, get_wind_input

from .base import Source

if TYPE_CHECKING:
    from ..means import MeanState

class NetworkSource(Source):
    def __init__(self):
        """
        At initialization, the network source must load the neural network it
        will use to adjust the launched ray volumes.
        """

        super.__init__()
        self._model = torch.jit.load(config.network_path)

    def _postprocess(
        self, *,
        n_step: int,
        mean: MeanState,
        data: np.ndarray,
        cdx: np.ndarray,
        **_
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        
        """

        rays = torch.as_tensor(data.T)
        u = get_wind_input(mean, n_step).expand(rays.shape[0], -1, -1)
        k, m = get_wavenumbers(u.numpy(), self._model(u, rays).numpy())
        dens = data[0] * data[-1] / k

        data[0] = k
        data[2] = m
        data[-1] = dens

        return data, cdx



