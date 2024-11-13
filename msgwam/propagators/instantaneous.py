from __future__ import annotations
from typing import TYPE_CHECKING, Self

import numpy as np

from .. import config
from ..dispersion import get_cg_r, get_omega_hat
from ..utils import shapiro_filter

from .base import Propagator
from .jitted import get_steady_action_fluxes

if TYPE_CHECKING:
    from ..means import MeanState

class InstantaneousPropagator(Propagator):
    """
    Class implementing the steady-state monochromatic gravity wave scheme from
    Section 3b of Bölöni et al. (2021).
    """

    def __init__(self, mean: MeanState) -> None:
        """
        At initialization, the `SteadyPropagator` just needs to queue the waves
        for the zeroth time step, since `get_fluxes` is called before `step`.
        """

        super().__init__(mean)
        self.step(mean, 0)
    
    def get_fluxes(self, mean: MeanState, net: bool=True) -> np.ndarray:
        """
        For the steady-state propagator, the action fluxes are propagated by the
        imported numba-compiled function. They are then multiplied by the
        appropriate wavenumbers to compute the requested momentum fluxes.
        """

        u = np.interp(mean.z_faces, mean.z_centers, mean.u)
        v = np.interp(mean.z_faces, mean.z_centers, mean.v)
        N = np.interp(mean.z_faces, mean.z_centers, mean.N)
        rho = np.interp(mean.z_faces, mean.z_centers, mean.rho)

        k, l, m, dk, dl, dm, dens = self._to_launch
        omega = get_omega_hat(k, l, m, N[0]) + k * u[0] + l * v[0]
        source_flux = get_cg_r(k, l, m, N[0]) * (dens * dk * dl * dm)

        args = [k, l, u, v, N, rho, omega, source_flux]
        action_flux = get_steady_action_fluxes(*args)

        if net:
            wvns = [k, l]

        else:
            wvns = [
                np.maximum(k, 0), np.minimum(k, 0),
                np.maximum(l, 0), np.minimum(l, 0)
            ]

        fluxes = np.vstack([(wvn * action_flux).sum(axis=1) for wvn in wvns])

        if config.shapiro_filter:
            fluxes[:, 1:-1] = shapiro_filter(fluxes.T).T

        return fluxes

    def step(self, mean: MeanState, n_step: int) -> Self:
        """
        The only time-dependent behavior of the steady-state scheme is the
        querying of the potentially time-variable source.
        """

        self._to_launch, _ = self._source.launch(mean, n_step)
        return self