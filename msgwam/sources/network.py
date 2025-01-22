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
    """
    This `Source` subclass can be used in two ways, based on the data that is
    passed via `config.network_path`. If that path points to a JITted pretrained
    neural network, then this source will use that network to predict adjusted
    values of k and m at launch time.

    However, if that path points to a numpy array, the values there will be used
    to replace the k and m values at launch time directly. This mode is useful
    for validating the `Adjuster` training dataset obtained through inversion.
    For this to work, the integration must be performed with the same settings
    as were used during data generation.
    """

    def __init__(self):
        """
        At initialization, the network source must load either the neural
        network or the array which will be used to adjust wavenumbers at launch.
        """

        super.__init__()

        if config.network_path.endswith('.jit'):
            self._model = torch.jit.load(config.network_path)

        elif config.network_path.endswith('.npy'):
            data = np.load(config.network_path)
            self._model = data.reshape(-1, config.n_source, 2)
            self._n_min = config.lookback // config.dt

        message = 'Unknown file type for NetworkSource'
        raise ValueError(f'{message}: {config.network_path}')

    def _postprocess(
        self, *,
        n_step: int,
        mean: MeanState,
        data: np.ndarray,
        cdx: np.ndarray,
        **_
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        If a network is in use, format the inputs appropriately and get the new
        k and m values. If a substitution array is being used, find the right
        index into that array. Either way, overwrite the returned wavenumbers
        and adjust the density to conserve momentum density.
        """

        if config.network_path.endswith('.jit'):
            rays = torch.as_tensor(data.T)
            u = get_wind_input(mean, n_step).expand(rays.shape[0], -1, -1)
            k, m = get_wavenumbers(u.numpy(), self._model(u, rays).numpy())

        elif config.network_path.endswith('npy'):
            if n_step - self._n_min < 0:
                return data, cdx
            
            k, m = self._model[n_step - self._n_min].T

        dens = data[0] * data[-1] / k
        data[0] = k
        data[2] = m
        data[-1] = dens

        return data, cdx



