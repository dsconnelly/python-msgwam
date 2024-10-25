from __future__ import annotations
from abc import abstractmethod
from typing import TYPE_CHECKING, Self

import numpy as np

from .. import config
from ..sources import Source
from ..utils import FactoryABC

if TYPE_CHECKING:
    from ..means import MeanState

class Propagator(FactoryABC):
    """
    Base class for schemes that take gravity wave source information and convert
    it to momentum flux profiles to drive the mean flow.
    """

    def __init__(self, mean: MeanState) -> None:
        """
        Initialize the propagator by creating a `Source` object that can be
        queried during integration to find out what waves to propagate.

        Parameters
        ----------
        mean
            Initial mean state of the system. Not used here, but can be used by
            subclasses and included here for the benefit of the type checker.

        """

        self._source = Source.from_name(config.source_type)

    @abstractmethod
    def get_fluxes(self, mean: MeanState, net: bool=True) -> np.ndarray:
        """
        Calculate gravity wave momentum fluxes at vertical grid cell faces.

        Parameters
        ----------
        mean
            Current mean state of the system.
        net
            If `True`, then the output array will have two rows corresponding to
            total zonal and meridional fluxes. If `False`, the output array will
            have four rows, corresponding to eastward, westward, northward, and
            southward fluxes. Note that the fluxes are signed even if `not net`,
            so that westward and southward fluxes are negative.

        Returns
        -------
        np.ndarray
            Two-dimensional array of momentum fluxes. The interpretation of the
            first dimension depends as described above by `net`, and the second
            dimension ranges over cell faces in the vertical grid.

        """
        ...

    @abstractmethod
    def step(self, mean: MeanState, n_step: int) -> Self:
        """
        Advance the propagator by one time step.

        Parameters
        ----------
        mean
            Current mean state of the system.
        n_step
            Index of the current time step.

        Returns
        -------
        Propagator
            Updated propagator.

        """
        ...