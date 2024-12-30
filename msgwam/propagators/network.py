from __future__ import annotations
from typing import TYPE_CHECKING, Self

import numpy as np
import torch

from msgwam.means import MeanState

from .. import config
from ..dispersion import get_cg_r
from ..utils import shapiro_filter

from .base import Propagator

if TYPE_CHECKING:
    from ..means import MeanState

class NetworkPropagator(Propagator):
    """Wave propagation using a trained neural network emulator."""

    def __init__(self, mean: MeanState) -> None:
        """
        
        """

        super().__init__(mean)
        self.model = torch.jit.load(config.network_path)
        
        self._n_ahead = int(config.time_horizon * 86400 / config.dt)
        self._forecast = np.zeros((2, self._n_ahead, config.n_grid))
        self._until_next = np.ones(self._source._data.shape[-1])

        self.step(mean, 0)

    def get_fluxes(self, _, net: bool=True) -> np.ndarray:
        """
        
        """

        zonal = self._forecast[:, 0]
        meridional = np.zeros((2, config.n_grid))

        if net:
            zonal = zonal.sum(axis=0)
            meridional = meridional.sum(axis=0)

        fluxes = np.vstack((zonal, meridional))
        if config.shapiro_filter:
            fluxes[:, 1:-1] = shapiro_filter(fluxes.T).T

        return fluxes

    def step(self, mean: MeanState, n_step: int) -> Self:
        """
        The only time-dependent behavior of the steady-state scheme is the
        querying of the potentially time-variable source.
        """

        self._forecast = np.roll(self._forecast, -1, axis=1)
        self._forecast[:, -1] = 0

        self._until_next -= 1
        cdx = self._until_next == 0
        to_launch, _ = self._source.launch(mean, n_step, cdx)

        k, l, m, *_ = to_launch
        cg_r = get_cg_r(k, l, m, mean.N[0])
        self._until_next[cdx] = np.ceil(config.dr_init / cg_r / config.dt)

        X = torch.as_tensor(to_launch.T)
        u = torch.as_tensor(mean.u[None]).expand(X.shape[0], -1)
        
        output = self.model(u, X).numpy()
        output = self._dimensionalize(to_launch, output)
        idxs = [to_launch[0] > 0, to_launch[0] < 0]

        for i, idx in enumerate(idxs):
            self._forecast[i] += output[idx].sum(axis=0)

        return self

    def _dimensionalize(self, X: np.ndarray, output: np.ndarray) -> np.ndarray:
        """
        
        """

        k, l, m, dk, dl, dm, dens = X
        cg_r = get_cg_r(k, l, m, config.N_ref)
        T = (config.z_max - config.z_min) / cg_r
        T = np.minimum(T, config.time_horizon * 86400)

        action = dens * dk * dl * dm
        factor = abs(k) * action * config.dr_init / T

        n_persist = np.round(T / config.dt).astype(int)
        weights = np.zeros((len(T), self._n_ahead))

        for j, n in enumerate(n_persist):
            weights[j, :n] = 1

        output = output * factor[:, None]
        return output[:, None] * weights[..., None]
