from __future__ import annotations
from typing import TYPE_CHECKING, Self

import numba as nb
import numpy as np
import torch

from msgwam import config
from msgwam.dispersion import get_omega_hat
from msgwam.propagators import Propagator

from .. import hyperparameters as hp

from .generation import get_pdx, project
from .training.transforms import apply_smoothing

if TYPE_CHECKING:
    from msgwam.means import MeanState

class NetworkPropagator(Propagator):
    def __init__(self, mean: MeanState):
        """
        Initializes arrays to hold the current bulk momentum profile in each
        quadrant and phase speed bin, as well as the most recently calculated
        momentum flux in each direction.
        """

        super().__init__(mean)

        self._model = torch.jit.load(config.model_path)
        self._M = np.zeros((4, config.n_bins, config.n_grid - 1))
        self._F = np.zeros((4, config.n_grid))
        self.step(mean, 0)

    def get_fluxes(self, _, net: bool=True) -> np.ndarray:
        """
        The `NetworkPropagator` does most of its work in `step`, including the
        calculation of the time-averaged flux profiles. Here all that needs to
        be done is add the relevant signed components of `net`.
        """

        signs = np.array([1, 1, -1, -1])
        F = signs[:, None] * self._F

        if net:
            return np.vstack((F[0] + F[2], F[1] + F[3]))
        
        return F[[0, 2, 1, 3]]

    def step(self, mean: MeanState, n_step: int) -> Self:
        """
        Adds the momentum flux associated with ray volumes that launch this time
        step to the momentum flux profiles, and then uses the loaded network to
        advance the state of the system.
        """

        n_seconds = config.dt * n_step
        if n_seconds % hp.generation.dt_output:
            return self
        
        C = self._make_C(mean)
        M = self._M + self._check_source(mean, n_step)
        budget = M.sum(axis=(1, 2), keepdims=True)
        M = M / budget

        C = np.hstack((C, np.log(budget[:, 0])))
        inputs = map(torch.as_tensor, [C, apply_smoothing(M)])
        F_v, F_h = [out.numpy() for out in self._model(*inputs)]
        dM, F = _get_dM_and_F(M, F_v, F_h)

        self._M = (M + dM) * budget
        self._F = F * budget[:, 0] * mean.dz / hp.generation.dt_output

        return self

    def _check_source(self, mean: MeanState, n_step: int) -> np.ndarray:
        """
        Get the extra momentum flux to add to each bin by checking the source
        and then allowing the waves to break according to the Lindzen criterion.
        """

        N = np.interp(config.r_source, mean.z_centers, mean.N)
        G2 = np.interp(config.r_source, mean.z_centers, mean.G2)

        (dr, k, l, m, dk, dl, dm, dens), _ = self._source.launch(mean, n_step)
        omega_hat = get_omega_hat(k, l, m, N, G2)
        cp_hat = omega_hat / abs(k + l)

        action = dens * dk * dl * dm
        r = config.r_source - 0.5 * dr
        pdx = get_pdx(k, l, cp_hat, config.n_bins)

        wvn_hor_sq = k ** 2 + l ** 2
        wvn_sq = wvn_hor_sq + m ** 2
        S = action * wvn_hor_sq * m ** 2 / (omega_hat * wvn_sq)
        P, Q = np.zeros((2, 4, config.n_grid - 1))

        threshold = mean.rho / 2
        project(r, dr, mean.z_faces, S, pdx // config.n_bins, P)
        project(r, dr, mean.z_faces, S * wvn_sq, pdx // config.n_bins, Q)
        P = P - threshold

        idx = Q != 0
        kappa = np.zeros_like(P)
        kappa[idx] = P[idx] / Q[idx]

        kappa = kappa.max(axis=1)[pdx // config.n_bins]
        factor = np.maximum(0, 1 - wvn_sq * kappa)
        mom = abs((k + l) * factor * action)

        out = np.zeros((4 * config.n_bins, config.n_grid - 1))
        project(r, dr, mean.z_faces, mom, pdx, out)

        return out.reshape(4, config.n_bins, config.n_grid - 1)

    def _make_C(self, mean: MeanState) -> torch.Tensor:
        """
        Assemble the input to the neural network consisting of the mean wind,
        buoyancy frequency, and latitude.
        """

        wind = np.vstack((mean.u, mean.v, -mean.u, -mean.v))
        N = np.vstack((mean.N, mean.N, mean.N, mean.N))
        lat = config.latitude * np.ones((4, 1))

        return torch.as_tensor(np.hstack((wind, N, lat)))

@nb.njit
def _get_dM_and_F(
    M: np.ndarray,
    F_v: np.ndarray,
    F_h: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Get appropriately clipped values for the momentum update `dM` and the bin-
    summed momentum flux to return to the propagator.

    Parameters
    ----------
    M
        Current bulk momentum state, as passed to the neural network.
    F
        Provisional fluxes as returned by the neural network (shape profiles
        scaled by amplitudes) to be possibly clipped.

    Returns
    -------
    torch.Tensor
        Tensor of changes in momentum in each cell.
    torch.Tensor
        Vertical momentum fluxes summed over all bins.

    """

    dM = np.zeros_like(M)
    for i in range(M.shape[0]):
        for k in range(M.shape[2]):
            F_bot = F_v[i, :, k].sum()
            F_top = F_v[i, :, k + 1].sum()
            deficit = F_top - F_h[i, 0, k] - (M[i, :, k].sum() + F_bot)

            if deficit > 1e-14:
                sink = min(F_h[i, 0, k] + deficit, 0)
                deficit = deficit + F_h[i, 0, k] - sink
                F_h[i, 0, k] = sink

            if deficit > 1e-14:
                factor = (F_top - deficit) / F_top
                F_v[i, :, k + 1] = factor * F_v[i, :, k + 1]

            for j in range(M.shape[1]):
                F_in = F_h[i, j, k] + F_v[i, j, k]
                F_out = F_h[i, j + 1, k] + F_v[i, j, k + 1]
                deficit = F_out - (M[i, j, k] + F_in)

                if deficit > 0:
                    sink = F_h[i, j + 1, k] - deficit
                    F_out = F_out - F_h[i, j + 1, k] + sink
                    F_h[i, j + 1, k] = sink

                dM[i, j, k] = F_in - F_out

    return dM, F_v.sum(axis=-2)
