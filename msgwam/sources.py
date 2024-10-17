from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import cftime
import numpy as np
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

        ds = get_spectrum()
        n_launch = int((86400 * config.n_day) // config.dt_launch) + 1

        if 'time' in ds.coords and n_launch > 1:
            seconds = config.dt_launch * torch.arange(n_launch)
            time = cftime.num2date(seconds, f'seconds since {EPOCH}')

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
    ) -> tuple[torch.Tensor, torch.Tensor]:
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

        In addition to the ray properties, implementations must return a tensor
        describing which of the requested slots each returned ray volume ought
        to replace. Most subclasses can just return `jdx` as is. However, some
        stochastic sources might wish to replace only a subset of the requested
        ray volumes and can indicate that with this return value.
        
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
        torch.Tensor
            Subset of `jdx` indicating the replaced ray volumes. Should have the
            same length as the second dimension of the first return value.

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

        if not config.rescale_fluxes or config.dt_launch == float('inf'):
            return torch.ones_like(data[-1])

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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """A `SimpleSource` just returns the source slots indexed by `jdx`."""

        return self.data[:, i * config.dt // config.dt_launch, jdx], jdx

class StochasticSource(SimpleSource):
    def __init__(self) -> None:
        """
        
        """

        super().__init__()
        self.cg_source = cg_r(*self.data[2:5])
        times = torch.ceil(self.data[1] / (self.cg_source * config.dt))
        self.launch_rate = config.epsilon * (1 / times).sum(dim=-1)

    def launch(
        self,
        i: int,
        jdx: torch.Tensor,
        _: MeanState
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        
        """

        weights = self.cg_source[i, jdx]
        weights = weights / weights.sum()
        weights = weights.numpy()

        size = int(torch.floor(self.launch_rate[i]))
        if torch.rand(1) < self.launch_rate[i] - size:
            size = size + 1

        size = min(size, len(jdx))
        jdx = jdx[np.random.choice(len(jdx), size, False, weights)]

        output, _ = super().launch(i, jdx, None)
        output[-1] = output[-1] / config.epsilon

        return output, jdx

class NetworkSource(SimpleSource):
    def __init__(self) -> None:
        """
        At initialization, a `NetworkSource` loads the JITted `CoarseNet` model
        specified in the config file.
        """

        super().__init__()
        self.model = torch.jit.load(config.model_path)

    def launch(
        self,        
        i: int,
        jdx: torch.Tensor,
        mean: MeanState
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        A `NetworkSource` uses the zonal mean profile included in `mean` to
        apply the `CoarseNet` to the desired ray volumes.
        """

        Y = super().launch(i, jdx, None)[0].T
        u = mean.u[None].expand(Y.shape[0], -1)

        with torch.no_grad():
            return self.model(u.double(), Y).T, jdx