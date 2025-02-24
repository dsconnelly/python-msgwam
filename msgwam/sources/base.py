from __future__ import annotations
from abc import abstractmethod
from typing import TYPE_CHECKING, Optional

import numpy as np

from .. import config
from ..dispersion import get_cg_r, get_dm, get_m
from ..utils import FactoryABC, cos_and_sin

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
        data = ds.to_array().values

        if data.ndim < 3:
            shape = (config.n_steps, *data.shape)
            data = np.broadcast_to(data, shape)

        else:
            data = data.transpose(1, 0, 2)

        self._data = data
        self._cp = ds['cp'].values
        self._phi = ds['phi'].values
        self._dc = np.diff(np.unique(self._cp))[0]

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
        and spectral densities that vary in time if the mean wind or buoyancy
        frequency are not constant. This function therefore also calculates
        those properties and includes them in the returned data.

        Because some sources (e.g. stochastic ones) might not return as many
        waves as were requested, we also return an array indicating which
        requested wave each returned wave corresponds to. The propagator is to
        interpret repeated indices in this array as indicating multiple copies
        of the ray volume in question, stacked in vertical space.

        Moreover, if `config.dt_launch` is greater than unity, the source is
        intermittent. This function therefore returns empty arrays if called
        at a non-integer multiple of the launch window.

        This is the public method meant to be called by propagators, and here we
        derive the time-varying wave properties mentioned above. We then rely on
        the `_postprocess` method implemented by subclasses to handle the launch
        logic particular to each source type.

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

        if n_step * config.dt % config.dt_launch != 0:
            return np.empty((7, 0)), np.empty(0, dtype=int)

        if cdx is None:
            cdx = np.arange(config.n_source)

        i = (n_step * config.dt) // config.dt_launch
        dk, dl, omega_hat, flux = self._data[i][:, cdx]
        cos, sin = cos_and_sin(self._phi[cdx])

        cp = self._cp[cdx]
        if config.extrinsic:
            u, v = mean.wind[:, 0]
            cp = cp - cos * u - sin * v

        wvn_hor = omega_hat / cp
        k, l = wvn_hor * cos, wvn_hor * sin

        m = get_m(k, l, omega_hat, mean.N[0])
        dm = get_dm(m, self._dc, mean.N[0])
        cg_r = get_cg_r(k, l, m, mean.N[0])

        dens = flux / abs(wvn_hor * dk * dl * dm * cg_r)
        data = np.vstack((k, l, m, dk, dl, dm, dens))

        return self._postprocess(
            n_step=n_step,
            mean=mean,
            data=data,
            cg_r=cg_r,
            cdx=cdx
        )

    @abstractmethod
    def _postprocess(
        self, *,
        n_step: int,
        mean: MeanState,
        data: np.ndarray,
        cg_r: np.ndarray,
        cdx: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply any source-specific logic to the selected wave properties. See the
        docstring for `launch` for more details on the return values. Subclass
        implementations can expect to have access to the parameters below.

        Parameters
        ----------
        n_step
            Index of the current time step.
        mean
            Current mean state of the system.
        data
            Array of properties of the ray volumes to launch.
        cg_r
            Already-calculated group velocities of ray volumes to launch.
        cdx
            Indices of the source channels selected for launch.

        """
