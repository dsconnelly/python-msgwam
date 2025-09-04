from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Self

import numpy as np
import xarray as xr

from scipy.linalg import lu_factor, lu_solve as _lu_solve
lu_solve = lambda A, b: _lu_solve(A, b.T).T

from .. import config
from ..utils import (
    get_bump,
    get_rho,
    get_time,
    get_vertical_grids,
    open_dataset
)

if TYPE_CHECKING:
    from ..propagators import Propagator

class MeanState:
    def __init__(self) -> None:
        """
        Initialize the mean state according to whether `config.tau_nudge` is
        zero (for a fully prescribed run), finite (for a run with nudging), or
        infinite (for a fully interactive run).
        """

        self.z_faces, self.z_centers = get_vertical_grids()
        self.dz: float = self.z_faces[1] - self.z_faces[0]
        
        self.is_prescribed = config.tau_nudge < np.inf
        self.is_interactive = config.tau_nudge > 0

        ds = None
        if self.is_prescribed:
            ds = open_dataset(config.prescribed_mean_file)
            coords = {'time' : get_time(), 'z_centers' : self.z_centers}
            ds = ds.interp(**coords, kwargs=dict(fill_value='extrapolate'))

        self.wind = self._init_wind(ds)
        self.rho, self.N, self.G2 = self._init_thermo(ds)

        if self.is_interactive:
            self._A, self._B = self._init_AB()
            self._last: list[np.ndarray] = []

    @property
    def nu(self) -> np.ndarray:
        """
        Compute the kinematic viscosity profile from the current mean state.

        Returns
        -------
        np.ndarray
            Kinematic viscosities at cell faces.
        
        """

        return np.interp(self.z_faces, self.z_centers, config.mu / self.rho)
    
    def step(self, prop: Propagator, n_step: int) -> Self:
        """
        Advance the mean state of the system by one time step.

        Parameters
        ----------
        prop
            Gravity wave propagator to use if momentum fluxes are calculated.
        n_step
            Index of the current time step.

        Returns
        -------
        MeanState
            Updated mean state of the system.

        """

        if not self.is_interactive:
            return self._update_prescribed(n_step)

        first_step = len(self._last) == 0
        if not first_step: last, dlast_dt = self._last

        dwind_dt = self._get_dwind_dt(prop, n_step)
        self.last = [self.wind, dwind_dt]

        if first_step:
            rhs = self.wind + config.dt * dwind_dt
            self.wind = lu_solve(self._A, rhs)

        else:
            rhs = 2 * self.wind - 0.5 * last
            rhs = rhs + config.dt * (2 * dwind_dt - dlast_dt)
            self.wind = lu_solve(self._B, rhs)

        return self._update_prescribed(n_step)
    
    @property
    def u(self) -> np.ndarray:
        """
        Return the zonal component of the mean wind.

        Returns
        -------
        np.ndarray
            Zonal wind velocities at cell centers.

        """

        return self.wind[0]
    
    @property
    def v(self) -> np.ndarray:
        """
        Return the meridional component of the mean wind.

        Returns
        -------
        np.ndarray
            Meridional wind velocities at cell centers.

        """

        return self.wind[1]
    
    def _get_dwind_dt(self, prop: Propagator, n_step: int) -> np.ndarray:
        """
        Calculate the mean wind tendency, which is given by minus the vertical
        derivative of the gravity wave momentum flux over density. An additional
        nudging term is added if `config.tau_nudge` is not infinite.

        Parameters
        ----------
        prop
            Gravity wave propagator to supply fluxes.
        n_step
            Index of the current time step.

        Returns
        -------
        np.ndarray
            Mean wind tendencies at cell centers.

        """

        flux_div = np.diff(prop.get_fluxes(self), axis=1) / self.dz
        dwind_dt = -flux_div / self.rho

        if self.is_prescribed:
            nudge = (self._wind[n_step] - self.wind) / config.tau_nudge
            dwind_dt = dwind_dt + nudge

        return dwind_dt

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

    def _init_thermo(
        self,
        ds: Optional[xr.Dataset]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Initialize thermodynamic variables.

        Parameters
        ----------
        ds
            `Dataset` from which to read and store time series for thermodynamic
            variables. If `None`, reasonable reference profiles will be used.

        Returns
        -------
        np.ndarray, np.ndarray, np.ndarray
            Arrays of density, buoyancy frequency, and squared scale height
            correction at vertical grid centers for the first time step.

        """

        if ds is None:
            rho = get_rho(self.z_centers)
            ones = np.ones_like(self.z_centers)
            G2 = ((1 / 2 - 2 / 7) / (2 * config.H_rho)) ** 2

            return rho, config.N_ref * ones, G2 * ones

        self._rho = ds['rho'].values
        self._N = np.sqrt(ds['N2'].values)
        self._G2 = ds['G2'].values

        return self._rho[0], self._N[0], self._G2[0]

    def _init_wind(self, ds: Optional[xr.Dataset]) -> np.ndarray:
        """
        Initialize mean wind profiles.

        Parameters
        ----------
        ds
            `Dataset` from which to read and store wind time series. If `None`,
            the run is fully interactive and a small perturbation is used.

        Returns
        -------
        np.ndarray
            Array whose first coordinate ranges over the zonal and meridional
            components of the mean wind at the first time step.

        """

        if ds is None:
            center = (config.z_min + config.z_max) / 2
            u = 10 * get_bump(self.z_centers, center, 10e3)
            v = 5 * get_bump(self.z_centers, center + 15e3, 10e3)
            v = v - 5 * get_bump(self.z_centers, center - 15e3, 10e3)

            return np.vstack((u, v))
        
        self._wind = np.stack((ds['u'], ds['v']), axis=1)
        return self._wind[0]

    def _update_prescribed(self, n_step: int) -> Self:
        """
        Update public-facing fields with values read in from disk, depending on
        whether the run is prescribed, interactive, or nudged.

        Parameters
        ----------
        n_step
            Time step to set values to.

        Returns
        -------
        Self
            Updated mean state of the system

        """

        names = ['rho', 'N', 'G2'] * (self.is_prescribed)
        names = names + (['wind'] * (not self.is_interactive))

        for name in names:
            setattr(self, name, getattr(self, f'_{name}')[n_step])

        return self
