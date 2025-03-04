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
        
        self.A, self.B = self._init_AB()
        self.grad_p = self._init_grad_p()
        self.last: list[np.ndarray] = []

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

        coriolis = config.f * np.vstack((self.v, -self.u))
        geo = (-self.grad_p / self.rho + coriolis) * config.geostrophic
        flux_div = np.diff(prop.get_fluxes(self), axis=1) / self.dz

        du_dz, dv_dz = np.diff(self.wind, axis=1) / self.dz
        du_dz = np.interp(self.z_centers, self.z_faces[1:-1], du_dz)
        dv_dz = np.interp(self.z_centers, self.z_faces[1:-1], dv_dz)
        upwelling = config.w_star * np.vstack((du_dz, dv_dz))

        return -flux_div / self.rho + geo - upwelling

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
    
    def _init_grad_p(self) -> np.ndarray:
        """
        Initialize the pressure gradients to be used in the geostrophic forcing,
        assuming the mean state is in balance at initialization.

        Returns
        -------
        np.ndarray
            Array of pressure gradients at vertical grid cell centers, whose two
            rows contain zonal and meridional gradients, respectively.

        """

        dp_dx = self.rho * config.f * self.v
        dp_dy = -self.rho * config.f * self.u

        return np.vstack((dp_dx, dp_dy))

    def _init_wind(self) -> np.ndarray:
        """
        Interactive runs start with a positive perturbation in the zonal wind
        field, and compensating positive and negative perturbations in the
        meridional wind field.      
        """

        center = (config.z_min + config.z_max) / 2
        u = 10 * _get_bump(self.z_centers, center, 10e3)
        v = 5 * _get_bump(self.z_centers, center + 15e3, 10e3)
        v = v - 5 * _get_bump(self.z_centers, center - 15e3, 10e3)

        return np.vstack((u, v))
    
def _get_bump(z: np.ndarray, center: float, width: float) -> np.ndarray:
    """
    Calculate a Gaussian bump of unit amplitude on the provided grid.

    Parameters
    ----------
    z
        Vertical grid points on which to compute the profile.
    center
        Location of the center of the bump.
    width
        Width of the bump.

    Returns
    -------
    np.ndarray
        Calculated bump profile. Has the same shape as `z`.

    """

    arg = (z - center) / width
    return np.exp(-0.5 * arg ** 2)