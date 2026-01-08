from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Self

import numpy as np
import xarray as xr

from scipy.linalg import lu_factor, lu_solve as _lu_solve
lu_solve = lambda A, b: _lu_solve(A, b.T).T

from .. import config
from ..utils import (
    gaussian_filter,
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

        self._stored: dict[str, np.ndarray] = {}
        self.wind, self.rho, self.N, self.G2 = self._init_state(ds)

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
            target = self._stored['wind'][n_step]
            nudge = (target - self.wind) / config.tau_nudge
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

    def _init_state(
        self, ds: Optional[xr.Dataset]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Initialize the wind and thermodynamic variables by reading from disc. If
        no dataset is provided, reasonable reference profiles are used instead.

        Parameters
        ----------
        ds
            Opened and interpolated dataset from which to read data.

        Returns
        -------
        np.ndarray, np.ndarray, np.ndarray, np.ndarray
            Wind, density, buoyancy frequency, and squared scale height
            correction profiles for the first time step at cell centers.

        """

        if ds is None:
            rho = get_rho(self.z_centers)
            ones = np.ones_like(self.z_centers)
            G2 = ones * ((1 / 2 - 2 / 7) / (2 * config.H_rho)) ** 2

            center = (config.z_min + config.z_max) / 2
            u = 10 * get_bump(self.z_centers, center, 10e3)
            v = 5 * get_bump(self.z_centers, center + 15e3, 10e3)
            v = v - 5 * get_bump(self.z_centers, center - 15e3, 10e3)

            return np.vstack((u, v)), rho, config.N_ref * ones, G2

        kwargs = {
            'seconds' : min(config.tau_nudge / 4, 86400),
            'z_centers' : 2 * self.dz * (config.tau_nudge > 0)
        }

        datas = []
        for name in ['u', 'v', 'rho', 'N2', 'G2']:
            data = np.sqrt(ds[name]) if name == 'N2' else ds[name]
            datas.append(gaussian_filter(data, **kwargs).values)

        u, v, *datas = datas
        self._stored['wind'] = np.stack((u, v), axis=1)
        for name, data in zip(['rho', 'N', 'G2'], datas):
            self._stored[name] = data

        return (data[0] for data in self._stored.values())

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
            setattr(self, name, self._stored[name][n_step])

        return self
