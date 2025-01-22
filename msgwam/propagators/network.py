from __future__ import annotations
from typing import TYPE_CHECKING, Self

import numpy as np
import torch

from .. import config
from ..dispersion import get_cg_r
from ..means import PrescribedWind
from ..utils import shapiro_filter, get_vertical_grids, get_wind_input
from .base import Propagator

if TYPE_CHECKING:
    from ..means import MeanState

class NetworkPropagator(Propagator):
    """Wave propagation using a trained neural network surrogate."""

    def __init__(self, mean: MeanState) -> None:
        """
        At initialization, the `NetworkPropagator` loads the neural network with
        which it will make predictions. It also creates `_forecast` array which
        will hold future predictions of time-averaged fluxes.
        """

        super().__init__(mean)

        self._model = torch.jit.load(config.network_path)
        self._n_ahead = int(max(config.time_horizon, config.dt) / config.dt)
        self._forecast = np.zeros((2, self._n_ahead, config.n_grid))
        self._until_next = np.ones(self._source._data.shape[-1])

        self.step(mean, 0)

    def get_fluxes(self, _, net: bool=True) -> np.ndarray:
        """
        To calculate the fluxes, the neural network simply needs to take the
        first time step of the `_forecast` array. The meridional fluxes are, for
        now, constrained to be zero everywhere.
        """

        zonal = self._forecast[:, 0]
        meridional = np.zeros_like(zonal)

        if net:
            zonal = zonal.sum(axis=0)
            meridional = meridional.sum(axis=0)

        fluxes = np.vstack((zonal, meridional))

        if config.shapiro_filter:
            fluxes[:, 1:-1] = shapiro_filter(fluxes.T).T

        return fluxes
    
    def step(self, mean: MeanState, n_step: int) -> Self:
        """
        Each time step, the `NetworkPropagator` must check the source and make
        predictions with the neural network for any newly-launched ray volumes.
        The `_forecast` array must also be rolled along its time dimension.
        """

        if not isinstance(mean, PrescribedWind):
            raise TypeError('NetworkPropagator requires a prescribed mean wind')

        self._until_next = self._until_next - 1
        self._forecast = np.roll(self._forecast, -1, axis=1)
        self._forecast[:, -1] = 0

        cdx = self._until_next == 0
        if cdx.sum() == 0:
            return self
        
        to_launch, _ = self._source.launch(mean, n_step, cdx)

        if config.dr_init < 0:
            self._until_next[cdx] = 1

        else:
            cg_r = get_cg_r(*to_launch[:3], mean.N[0])
            p, n_steps = np.modf(config.dr_init / cg_r / config.dt)
            n_steps[np.random.rand(len(n_steps)) < p] += 1
            self._until_next[cdx] = np.maximum(1, n_steps)

        rays = torch.as_tensor(to_launch.T)
        u = get_wind_input(mean, n_step).expand(rays.shape[0], -1, -1)
        output = self._dimensionalize(to_launch, self._model(u, rays).numpy())
        idxs = [to_launch[0] > 0, to_launch[0] < 0]

        for k, idx in enumerate(idxs):
            self._forecast[k] += output[idx].sum(axis=0)

        return self
        
    def _dimensionalize(
        self,
        to_launch: np.ndarray,
        output: np.ndarray
    ) -> np.ndarray:
        """
        Dimensionalize neural network outputs so that they can be returned to
        the integrator. Also adds a time dimension to the outputs, with profiles
        persisting for a duration determined by the source group velocity and by
        `config.time_horizon`.

        Parameters
        ----------
        to_launch
            Array of source ray volume information, as returned by the `launch`
            method of a `Source` object.
        output
            Nondimensional neural network output for each volume in `to_launch`.

        Returns
        -------
        np.ndarray
            Array of dimensionalized flux profiles whose first dimension ranges
            over individual ray packets, whose second dimdnesion ranges over
            time steps, and whose third dimension ranges over grid points.

        """

        k, l, m, dk, dl, dm, dens = to_launch
        cg_r = get_cg_r(k, l, m, config.N_ref)
        action = dens * (dk * dl * dm)

        if config.dr_init < 0:
            T = config.dt * np.ones_like(cg_r)
            factor = abs(k) * action * cg_r

        else:
            broken = abs(output) < 0.3
            z, _ = get_vertical_grids()

            z_break = z[np.argmax(broken, axis=1)]
            z_break[broken.sum(axis=1) == 0] = config.z_max
            z_break = np.maximum(z_break, 20e3)

            T = (z_break - config.z_min) / cg_r
            T = np.minimum(T, config.time_horizon)
            factor = abs(k) * action * config.dr_init / T

        weights = np.zeros((len(T), self._n_ahead))
        n_persist = np.round(T / config.dt).astype(int)
        for i, n in enumerate(n_persist):
            weights[i, :n] = 1

        return (output * factor[:, None])[:, None] * weights[..., None]
