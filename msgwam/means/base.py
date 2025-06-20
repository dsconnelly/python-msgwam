from __future__ import annotations
from abc import abstractmethod
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Generic, Self, TypeVar

import numpy as np

from .. import config
from ..utils import FactoryABC, get_vertical_grids

if TYPE_CHECKING:
    from ..propagators import Propagator

T = TypeVar('T')

class MeanState(FactoryABC, Generic[T]):
    def __init__(self) -> None:
        """
        Initialize the mean state of the model. Calls several abstract methods
        with context supplied by `_get_init_context`.
        """

        self.z_faces, self.z_centers = get_vertical_grids()
        self.dz: float = self.z_faces[1] - self.z_faces[0]

        with self._get_init_context() as context:
            self.rho, self.N, self.G2 = self._init_thermo(context)
            self.wind = self._init_wind(context)

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

    @abstractmethod
    def _get_init_context(self) -> AbstractContextManager[T]:
        """
        Get a context manager to be passed to the initialization functions. Can
        be `nullcontext`, if the state is initialized from nothing, but can also
        be used to supply an open file to subclasses that read in state data.

        Returns
        -------
        AbstractContextManager[T]
            Context to be used during state initialization.

        """
        ...

    @abstractmethod
    def _init_thermo(
        self,
        context: T
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Initialize the thermodynamic state variables.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            Initial profiles for density, buoyancy frequency, and squared scale
            height correction at cell centers.

        """
        ...

    @abstractmethod
    def _init_wind(self, context: T) -> np.ndarray:
        """
        Initialize the mean wind.

        Returns
        -------
        np.ndarray
            Array whose first and second rows contain the initial zonal and 
            meridional velocities, respectively, at cell centers.

        """
        ...
