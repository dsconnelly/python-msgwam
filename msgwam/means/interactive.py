from __future__ import annotations
from typing import TYPE_CHECKING, Self

import numpy as np

from scipy.linalg import lu_factor, lu_solve as _lu_solve
lu_solve = lambda A, b: _lu_solve(A, b.T).T

from .. import config

from .base import MeanState

if TYPE_CHECKING:
    from ..propagators import Propagator

class InteractiveWind(MeanState):
    def __init__(self) -> None:
        """
        The interactive mean also initializes an array to hold the last wind
        and tendency profiles (for SBDF2 time stepping) and LU factorizations of
        the matrices for backward Euler and SBDF2 time steps.
        """

        super().__init__()
        self.last: list[np.ndarray] = []
        self.A, self.B = self._init_AB()

    def step(self, prop: Propagator, _) -> Self:
        """
        Take a backward Euler step if this is the first step of the integration.
        Otherwise, use SBDF2. See Wang and Ruuth (2008) for details.
        """

        first_step = len(self.last) == 0
        if not first_step:
            last, dlast_dt = self.last

        dwind_dt = self._get_dwind_dt(prop)
        self.last = [self.wind, dwind_dt]

        if first_step:
            self.wind = lu_solve(self.A, self.wind + config.dt * dwind_dt)

        else:
            rhs = 2 * self.wind - 0.5 * last
            rhs = rhs + config.dt * (2 * dwind_dt - dlast_dt)
            self.wind = lu_solve(self.B, rhs)

        return self

    def _get_dwind_dt(self, prop: Propagator) -> np.ndarray:
        """
        Calculate the mean wind tendency, which is given by minus the vertical
        derivative of the gravity wave momentum flux over density.

        Parameters
        ----------
        prop
            Gravity wave propagator to supply fluxes.

        Returns
        -------
        np.ndarray
            Mean wind tendencies at cell centers.

        """

        flux_div = np.diff(prop.get_fluxes(self), axis=1) / self.dz
        return -flux_div / self.rho

    def _init_AB(self) -> tuple[tuple, tuple]:
        """
        Interactive time stepping requires LU decompositions of matrices for
        backward Euler and SBDF2 steps. Since these matrices are constant in
        time, we precompute them once and save them.

        Returns
        -------
        tuple
            LU factorization of the backward Euler matrix.
        tuple
            LU factorizaiton of the SBDF2 matrix.

        """

        off_diag = self.nu[1:-1]
        diag = -(self.nu[:-1] + self.nu[1:])
        D = np.diag(diag) + np.diag(off_diag, 1) + np.diag(off_diag, -1)

        D[0, 0] = D[0, 0] - self.nu[0]
        D[-1, -1] = D[-1, -1] - self.nu[-1]
        D = D / (self.dz ** 2)

        m, _ = D.shape
        A = lu_factor(np.eye(m) - config.dt * D)
        B = lu_factor(3 * np.eye(m) / 2 - config.dt * D)

        return A, B

    def _init_wind(self) -> np.ndarray:
        """
        Interactive runs start with a small perturbation in the interior of the
        domain, so that the wind has some asymmetry.        
        """

        u = 0.25 * np.exp(-0.5 * ((self.z_centers - 50e3) / 3e3) ** 2)
        return np.vstack((u, np.zeros_like(u)))