from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import cftime
import torch
import xarray as xr

from . import config
from .constants import EPOCH
from .dispersion import cg_r
from .spectra import get_flux, get_spectrum

if TYPE_CHECKING:
    from .mean import MeanState

class Source(ABC):
    def __init__(self) -> None:
        """
        Initialize the source with data from the spectrum type specified in the
        configuration file. At initialization, the `Source` takes a `Dataset`
        of source spectrum information and matches it to the correct time grid.
        It then rescales the flux associated with each source ray volume to
        account for intermittency effects.
        """

        n_launch = (86400 * config.n_day) // config.dt_launch + 1
        seconds = config.dt_launch * torch.arange(n_launch)
        time = cftime.num2date(seconds, f'seconds since {EPOCH}')

        ds = get_spectrum()
        if 'time' in ds.coords:
            ds = self._average_backwards(ds)
            ds = ds.sel(time=time, method='ffill')

        data = torch.as_tensor(ds.to_array().values)
        data[-1] *= self._get_intermittency_factors(data)

        if data.dim() < 3:
            data = data.unsqueeze(1).expand(-1, n_launch, -1)

        self.data = data

    @property
    def n_slots(self) -> int:
        """Returns the number of phase speeds in the source spectrum."""

        return self.data.shape[-1]
    
    @abstractmethod
    def launch(
        self,        
        i: int,
        jdx: torch.Tensor,
        mean: MeanState
    ) -> torch.Tensor:
        """
        Each spectrum is discretized into a fixed number of slots corresponding
        to the last dimension of `self.data`. This function takes in the current
        time step and a tensor containing the slots of rays that have cleared
        the launch level, and returns the properties of the ray volumes that
        should be launched.

        A simple implementation of this method might return the slots as indexed
        by `jdx`. Subclasses, however, might implement more complex behavior.
        For example, a neural network source might modify the source data before
        returning it. For this reason, this method also accepts the current mean
        state of the system in case subclasses use the mean wind fields.
        
        Parameters
        ----------
        i
            Index of current time step.
        jdx
            Tensor of slots of ray volumes that have cleared the launch level.

        Returns
        -------
        torch.Tensor
            Tensor of properties of the ray volumes that should be launched.

        """
        ...

    @staticmethod
    def _average_backwards(ds: xr.Dataset) -> xr.Dataset:
        """
        If `config.dt_launch` is longer than the time step in the given source
        spectrum, this function averages the ray volume properties over the
        appropriate interval preceding the launch time. The spectral density is
        then adjusted so that the average flux is correct.
        
        If `config.dt_launch` is shorter than the source spectrum time step (for
        example, if it is equal to the integration time step) then this function
        leaves the dataset unchanged.

        Parameters
        ----------
        ds
            Dataset of time-varying source ray volume properties.

        Returns
        -------
        xr.Dataset
            Backwards-averaged dataset.

        """

        bins = ds['time'].dt.ceil(f'{config.dt_launch}S')
        targets = get_flux(ds).groupby(bins).mean()
        ds = ds.groupby(bins).mean()

        cs = ds['cp_x'].values
        dc = abs(cs[1] - cs[0])
        k, dk, dl = ds[['k', 'dk', 'dl']].to_array()
        ds['dens'] = abs(targets) / (k ** 2 * dk * dl * dc)

        return ds.rename(ceil='time')

    @staticmethod
    def _get_intermittency_factors(data: torch.Tensor) -> torch.Tensor:
        """
        Compute factors to scale the spectral density to account for rays that
        ought to launch more than once in a launch window. This rescaling is
        mainly to account for intermittent sources, where ray volumes might have
        group velocities implying many launches per window, but fast-moving rays
        can have their densities adjusted even when the source is constant.

        Parameters
        ----------
        data
            Tensor of ray volume properties whose first dimension corresponds to
            individual ray properties and whose last dimension corresponds to
            phase speeds. `data` may also have a middle time dimension.

        Returns
        -------
        torch.Tensor
            Tensor with the same shape as `data[-1]` containing the factors by
            which the spectral density should be scaled.

        """

        _, dr, k, l, m, *_ = data
        factors = cg_r(k, l, m) * config.dt_launch / dr
        factors = torch.clamp(factors, torch.ones_like(factors))

        return factors

class SimpleSource(Source):
    def launch(
        self,        
        i: int,
        jdx: torch.Tensor,
        _: MeanState
    ) -> torch.Tensor:
        """A `SimpleSource` just returns the source slots indexed by `jdx`."""

        return self.data[:, i * config.dt // config.dt_launch, jdx]