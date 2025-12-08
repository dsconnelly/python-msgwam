from __future__ import annotations
from typing import TYPE_CHECKING, Self

import numpy as np

from msgwam import config
from msgwam.dispersion import get_omega_hat
from msgwam.propagators import Propagator

from .utils import get_cg_r, get_qdx, get_transports, project, recover_omega_hat

if TYPE_CHECKING:
    from msgwam.means import MeanState

class EulerianPropagator(Propagator):
    def __init__(self, mean: MeanState):
        """
        Initialize the Eulerian propagator by defining the wavenumber and phase
        speed grids, and creating arrays to hold the momentum state.
        """

        super().__init__(mean)

        self._init_edges()        
        wvn = (self._edges_wvn[:-1] + self._edges_wvn[1:]) / 2
        cpt = (self._edges_cpt[:-1] + self._edges_cpt[1:]) / 2

        self._wvn = wvn[:, None, None]
        self._cpt = cpt[:, None]

        self._M = np.zeros((4, config.n_k, config.n_c, config.n_grid - 1))
        self._F = np.zeros((4, config.n_grid))
        self.step(mean, 0)

    def get_fluxes(self, _, net: bool=True) -> np.ndarray:
        """
        Return the fluxes stored during the most recent step.
        """

        signs = np.array([1, 1, -1, -1])
        F = signs[:, None] * self._F

        if net:
            return np.vstack((F[0] + F[2], F[1] + F[3]))
        
        return F[[0, 2, 1, 3]]
        
    def step(self, mean: MeanState, n_step: int) -> Self:
        """
        Take a step by advecting the bulk momentum according to the group
        velocity and then applying the breaking criterion.
        """

        M = self._M + self._check_source(mean, n_step)
        wind = np.vstack((mean.u, mean.v, -mean.u, -mean.v))
        N = np.interp(mean.z_faces, mean.z_centers, mean.N)
        N = np.vstack((N, N, N, N))[:, None, None]

        dt_o_dz = config.dt / mean.dz
        cg = get_cg_r(self._wvn, self._cpt, N, config.f)[..., 1:]
        phi_in, phi_out = get_transports(M, cg, self._edges_cpt, wind, dt_o_dz)
        
        dM = phi_in - phi_out 
        self._M = np.maximum(M + dM, 0)

        D = self._get_sinks(mean)
        self._M = self._M - D
        self._M[self._M < 1e-8] = 0

        F = phi_out / dt_o_dz
        self._F[..., 1:] = F.sum((1, 2))
        self._cache = [M, cg * M]

        return self

    @staticmethod
    def _allocate_bins(n: int, frac: float) -> np.ndarray:
        """
        Allocate a certain number of bins to the interval [0, 100], with more
        possibly concentrated in one half or the other.

        Parameters
        ----------
        n
            How many bins to allocate.
        frac
            Fraction of those bins that should be used on [0, 50].

        Returns
        -------
        np.ndarray
            Array of bin edges.

        """

        n_inner = max(int(frac * n), 1)
        edges = np.linspace(0, 50, n_inner + 1)
        n_outer = n - n_inner

        if n_outer > 0:
            outer = np.linspace(50, 100, n_outer + 1)[1:]
            edges = np.concatenate((edges, outer))

        return edges
    
    def _check_source(self, mean: MeanState, n_step: int) -> np.ndarray:
        """
        Check the propagator's source, project the new ray volumes onto the
        vertical and phase speed grids, and adjust the momentum state.
        """

        N = np.interp(config.r_source, mean.z_centers, mean.N)
        G2 = np.interp(config.r_source, mean.z_centers, mean.G2)
        (dr, k, l, m, dk, dl, dm, dens), _ = self._source.launch(mean, n_step)

        r_lo = config.r_source - dr
        r_hi = config.r_source * np.ones_like(dr)

        wvn = abs(k + l)
        mom = wvn * (dens * dk * dl * dm)

        omega_hat_lo = get_omega_hat(k, l, m - 0.5 * dm, N, G2)
        omega_hat_hi = get_omega_hat(k, l, m + 0.5 * dm, N, G2)
        c_lo = (omega_hat_lo - abs(config.f)) / wvn
        c_hi = (omega_hat_hi - abs(config.f)) / wvn

        qdx = get_qdx(k, l)
        pairs = (self._edges_wvn[:-1], self._edges_wvn[1:])
        out = np.zeros((4, config.n_k, config.n_c, config.n_grid - 1))
        wvn = np.clip(wvn, self._edges_wvn.min(), self._edges_wvn.max())

        for i, (lo, hi) in enumerate(zip(*pairs)):
            wdx = (lo <= wvn) & (wvn < hi)

            project(
                mom[wdx],
                c_lo[wdx], c_hi[wdx],
                r_lo[wdx], r_hi[wdx],
                self._edges_cpt,
                mean.z_faces,
                qdx[wdx],
                out[:, i]
            )

        return out

    def _get_sinks(self, mean: MeanState) -> np.ndarray:
        """
        Determine where polychromatic breaking should occur and reduce the bulk
        momentum density in those cells accordingly.
        """

        N2 = mean.N ** 2
        omega_hat = recover_omega_hat(self._wvn, self._cpt, mean.N, config.f)
        omega_hat_sq = omega_hat ** 2
        wvn_hor_sq = self._wvn ** 2

        m_sq = wvn_hor_sq * (N2 - omega_hat_sq) / (omega_hat_sq - config.f ** 2)
        Q = self._M * self._wvn * m_sq / omega_hat
        P = Q / (wvn_hor_sq + m_sq)

        P = P.sum((0, 1, 2))
        Q = Q.sum((0, 1, 2))
        P = P - mean.rho / 2

        idx = Q != 0
        kappa = np.zeros_like(P)
        kappa[idx] = P[idx] / Q[idx]

        factor = np.clip(1 - (wvn_hor_sq + m_sq) * kappa, 0, 1)
        return self._M * (1 - factor)

    def _init_edges(self) -> None:
        """
        Initialize the phase speed and wavenumber grids.
        """

        frac = 0.8 if config.extrinsic else 0
        self._edges_cpt = self._allocate_bins(config.n_c, 0.8)
        edges_wvl = self._allocate_bins(config.n_k, frac)

        edges_wvl[0] = 0.01 * edges_wvl[1]
        edges_wvl = config.T_hat_source * edges_wvl
        self._edges_wvn = 2 * np.pi / edges_wvl[::-1]
