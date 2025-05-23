from __future__ import annotations
from abc import abstractmethod
from typing import TYPE_CHECKING, Self

import numpy as np

from .. import config
from ..utils import FactoryABC, get_rho, get_vertical_grids, make_colored_noise

if TYPE_CHECKING:
    from ..propagators import Propagator

_SEED = 111

class MeanState(FactoryABC):
    def __init__(self) -> None:
        """
        Initialize the mean state of the model. Defines the faces and centers of
        the vertical grid, and then initializes the background density, buoyancy
        frequency, and kinematic viscosity profiles. Initializes the mean wind
        by calling a function that should be implemented by subclasses.
        """

        self.z_faces, self.z_centers = get_vertical_grids()
        self.dz: float = self.z_faces[1] - self.z_faces[0]

        self.N = self._init_N()
        self.rho = get_rho(self.z_centers)
        self.wind = self._init_wind()
        self.nu = self._init_nu()

    @abstractmethod
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
        ...

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

    def _init_N(self) -> np.ndarray:
        """
        Initialize the background buoyancy frequency profile.

        Returns
        -------
        np.ndarray
            Buoyancy frequencies at cell centers.

        """

        z_mid = (config.z_min + config.z_max) / 2
        t = np.tanh((self.z_centers - z_mid) / config.H_N)
        a = config.N_ref_max - config.N_ref_min
        N = config.N_ref_min + a * (t + 1) / 2

        rng = np.random.default_rng(_SEED)
        scales = [config.H_N, 0.25 * config.H_N]
        noise = make_colored_noise(self.z_centers, *scales, rng=rng)

        return N + config.N_ref_noise * noise

    def _init_nu(self) -> np.ndarray:
        """
        Initialize the kinematic viscosity profile.

        Returns
        -------
        np.ndarray
            Kinematic viscosities at cell faces.

        """

        return np.interp(self.z_faces, self.z_centers, config.mu / self.rho)

    @abstractmethod
    def _init_wind(self) -> np.ndarray:
        """
        Initialize the mean wind.

        Returns
        -------
        np.ndarray
            Array whose first and second rows contain the zonal and meridional
            velocities, respectively, at cell centers.

        """
        ...