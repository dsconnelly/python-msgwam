from typing import Self

import numpy as np

from .. import config
from ..utils import get_time, open_dataset

from .base import MeanState

class PrescribedWind(MeanState):
    def step(self, _, n_step: int) -> Self:
        """
        When the mean flow is prescribed, `step` just needs to set `self.wind`
        to point to the appropriate entry in the loaded time series.
        """

        self.wind = self._wind[n_step]
        return self

    def _init_wind(self) -> np.ndarray:
        """
        Initializes the mean wind by interpolating a dataset loaded from disc to
        the appropriate time and z values. Stores the dataset for later updates.
        """

        with open_dataset(config.prescribed_wind_file) as ds:
            ds = ds.interp(time=get_time(), z_centers=self.z_centers)
            self._wind = np.stack((ds['u'], ds['v']), axis=1)

        return self._wind[0]
        