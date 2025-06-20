from typing import Self

import numpy as np
import xarray as xr

from .. import config
from ..utils import get_time, open_dataset

from .base import MeanState

class PrescribedWind(MeanState[xr.Dataset]):
    def step(self, _, n_step: int) -> Self:
        """
        When the mean flow is prescribed, `step` just needs to set the mean wind
        and the thermodynamic variables to the appropriate point in the series.
        """

        self.wind = self._wind[n_step]

        self.rho = self._rho[n_step]
        self.N = self._N[n_step]
        self.G2 = self._G2[n_step]

        return self

    def _get_init_context(self) -> xr.Dataset:
        """
        Open the dataset specified in the configuration file and interpolate it
        to the appropriate time and `z` values.
        """

        ds = open_dataset(config.prescribed_mean_file)
        coords = {'time' : get_time(), 'z_centers' : self.z_centers}
        kwargs = {'fill_value' : 'extrapolate'}

        return ds.interp(**coords, kwargs=kwargs)
    
    def _init_wind(self, context: xr.Dataset) -> np.ndarray:
        """Save the mean wind time series for future use."""

        self._wind = np.stack((context['u'], context['v']), axis=1)
        return self._wind[0]
        
    def _init_thermo(
        self,
        context: xr.Dataset
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Saves the thermodynamic variable time series for future use."""
        
        self._rho = context['rho'].values
        self._N = np.sqrt(context['N2'].values)
        self._G2 = context['G2'].values

        return self._rho[0], self._N[0], self._G2[0]
