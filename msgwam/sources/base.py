from __future__ import annotations
from abc import abstractmethod
from typing import TYPE_CHECKING, Optional

import numpy as np

from .. import config
from ..dispersion import get_cg_r, get_dm, get_m
from ..utils import FactoryABC, get_time

from .spectra import get_spectrum

if TYPE_CHECKING:
    from ..means import MeanState

class Source(FactoryABC):
    """
    Sources are responsible for providing the propagator with the properties of
    the waves that should be launched each time step.
    """

    def __init__(self) -> None:
        """
        Initialize the source by storing the spectral data. If the spectrum is
        constant in time, a dummy dimension is added for later consistency.
        """

        ds = get_spectrum()
        if 'time' in ds.coords:
            ds = ds.sel(time=get_time(), method='ffill')

        self._cp_x = ds['cp_x'].values
        self.dc = self._cp_x[1] - self._cp_x[0]
        data = ds.to_array().values

        if data.ndim < 3:
            shape = (config.n_steps, *data.shape)
            data = np.broadcast_to(data, shape)

        self._data = data

    def launch(
        self,
        mean: MeanState,
        n_step: int,
        cdx: Optional[np.ndarray]=None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Each source spectrum is discretized into `config.n_source` spectral
        elements, whose properties may vary in time. This function takes the
        curent time step, the current mean state of the system, and an array
        indexing those spectral elements, and returns the properties of the
        waves that should be launched.

        Even constant-in-time spectra may have vertical wavenumbers, extents,
        and spectral densities that vary in time if the buoyancy frequency is
        not constant. This function therefore also calculates those properties
        and includes them in the returned data.

        Because some sources (e.g. stochastic ones) might not return as many
        waves as were requested, we also return an array indicating which
        requested wave each returned wave corresponds to.

        This is the public method meant to be called by propagators. This method
        relies on the `_launch` method implemented by subclasses, which handles
        the actual launch logic particular to each source. This method than
        calculates the additional wave properties mentioned above.

        Parameters
        ----------
        mean
            Current mean state of the system.
        n_step
            Index of the current time step.
        cdx
            Indices of the requested spectral elements in the source. If `None`,
            the entire source spectrum will be requested.

        Returns
        -------
        np.ndarray
            Array whose first dimension ranges over ray volume properties and
            whose second dimension ranges over waves to be launched.
        np.ndarray
            Subset of `cdx` indicating which requested wave each column of the
            first returned array corresponds to.

        """

        if cdx is None:
            cdx = np.arange(config.n_source)

        (k, l, dk, dl, flux), cdx = self._launch(mean, n_step, cdx)
        m = get_m(k, l, self._cp_x[cdx], mean.N[0])
        dm = get_dm(m, self.dc, mean.N[0])

        cg_r = get_cg_r(k, l, m, mean.N[0])
        dens = flux / abs(k * dk * dl * dm * cg_r)
        data = np.vstack((k, l, m, dk, dl, dm, dens))

        return data, cdx

    @abstractmethod
    def _launch(
        self,
        mean: MeanState,
        n_step: int,
        cdx: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Subclass-specific source logic called by `launch`. See the docstring for
        that method for more details on parameters and return values.
        """
        ...

