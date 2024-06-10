from abc import ABC, abstractmethod

import torch

from . import config
from .dispersion import cg_r, cp_x, m_from
from .spectra import get_spectrum

class Source(ABC):
    def __init__(self) -> None:
        """
        Initialize the source with data from the spectrum type specified in the
        config file. That parameter must correspond to the name of a function
        defined in spectra.py.

        """

        self.data = get_spectrum()

    @abstractmethod
    def launch(
        self,
        jdx: torch.Tensor,
        u: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Each spectrum is discretized into a fixed number of slots, each of which
        corresponds to a column in `self.data`. This function takes in a tensor
        containing the slots of rays that have cleared the launch level, along
        with the current state of the mean zonal wind, and returns the 
        properties of the ray volumes that should be launched.

        This function may simply return the columns of `self.data` indexed by
        `jdx`. However, subclasses might implement more complex behavior. For
        example, a stochastic source might return only a subset of the requested
        slots, while a neural network source might modify the source data before
        returning it.

        In addition to the properties of the ray volumes to launch, the source
        returns a helper tensor containing the slots that are replaced by each
        of the returned ray volumes.

        Parameters
        ----------
        jdx
            Tensor of slots of ray volumes that have cleared the launch level.
        u
            Current zonal wind profile (may be ignored by subclasses).

        Returns
        -------
        torch.Tensor
            Tensor of properties of the ray volumes that should be launched.
        torch.Tensor
            Tensor of slots that are replaced by one of the rays in the first
            output tensor.

        """
        ...

    @property
    def n_slots(self) -> int:
        """
        Count the total number of ray volumes in this source. Implemented as a
        property so that subclasses which modify the source data can do so after
        `__init__` without having to redeclare an instance field.

        Returns
        -------
        int
            Number of ray volumes in the source spectrum.

        """

        return self.data.shape[1]

class DeterministicSource(Source):
    def launch(self, jdx: torch.Tensor, _) -> tuple[torch.Tensor, torch.Tensor]:
        """A `DeterministicSource` simply returns the requested columns."""

        return self.data[:, jdx], jdx

class CoarseSource(DeterministicSource):
    def __init__(self) -> None:
        """
        At initialization, a CoarseSource replaces the loaded source spectrum
        with an appropriately coarsened version.
        """

        super().__init__()
        self.data = self.coarsen(self.data)

    @classmethod
    def coarsen(cls, spectrum: torch.Tensor) -> torch.Tensor:
        """
        Coarsen a source spectrum according to the loaded `config` settings.

        Parameters
        ----------
        spectrum
            Tensor of source ray volumes whose first dimension ranges over ray
            volume properties and whose second dimension ranges over rays in the
            source spectrum.

        Returns
        -------
        torch.Tensor
            Tensor of shape `(9, spectrum.shape[1] // config.coarse_width)` of
            coarsened ray volumes.

        """

        cs = cp_x(*spectrum[2:5])
        cs = cs.reshape(-1, config.coarse_width)
        
        c_lo, _ = cs.min(dim=-1)
        c_hi, _ = cs.max(dim=-1)
        cs = (c_lo + c_hi) / 2

        spectrum = spectrum.reshape(9, -1, config.coarse_width)
        fluxes = cls._get_fluxes(spectrum).sum(dim=-1)
        spectrum = spectrum.mean(dim=-1)

        r, dr = spectrum[:2]
        spectrum[0] = r + (1 - config.coarse_height) * dr / 2
        spectrum[1] = dr * config.coarse_height

        k, l = spectrum[2:4]
        spectrum[4] = m_from(k, l, cs)

        if config.coarse_width > 1:
            dc = (c_hi - c_lo) / (1 - 1 / config.coarse_width)
            spectrum[7] = abs(k * dc / cg_r(k, l, spectrum[4]))

        spectrum[8] *= fluxes / cls._get_fluxes(spectrum)

        return spectrum

    @staticmethod
    def _get_fluxes(spectrum: torch.Tensor) -> torch.Tensor:
        """
        Calculate the pseudomomentum fluxes associated with ray volumes in a
        source spectrum.

        Parameters
        ----------
        spectrum
            Tensor of source ray volumes. The first dimension must range over
            ray properties, and the other dimensions may be arbitrary.

        Returns
        -------
        torch.Tensor
            Tensor of shape `spectrum.shape[1:]` containing the pseudomomentum
            flux associated with each ray volume.

        """

        k, l, m, dk, dl, dm, dens = spectrum[2:]
        return k * cg_r(k, l, m) * dens * dk * dl * dm
    
class NetworkSource(CoarseSource):
    def __init__(self) -> None:
        """
        At initialization, a `CoarseSource` loads the jitted model saved at the
        location indicated by the config file.
        """

        super().__init__()
        self.model = torch.jit.load(config.model_path)

    def launch(
        self,
        jdx: torch.Tensor,
        u: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        A `NetworkSource` lets its loaded network modify the coarsened ray
        volumes before sending them back to the integrator.
        """

        return self.model(u, self.data[:, jdx]), jdx

class StochasticSource(Source):
    def __init__(self) -> None:
        """
        At initialization, the stochastic source calculates the vertical group
        velocity of each ray volume in the spectrum and thereby estimates the
        launch rate that would be attained without stochasticity.
        """

        super().__init__()

        self.cg_source = cg_r(*self.data[2:5])
        times = torch.ceil(config.dr_init / (self.cg_source * config.dt))
        self.launch_rate = config.epsilon * (1 / times).sum()

    def launch(self, jdx: torch.Tensor, _) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return `config.epsilon` times the number of rays requested, sampled such
        that faster ray volumes are launched more frequently.
        """

        size = int(torch.floor(self.launch_rate))
        if torch.rand(1) < self.launch_rate - size:
            size = size + 1

        weights = self.cg_source[jdx]
        weights = weights / weights.sum()
        jdx = jdx[torch.multinomial(weights, num_samples=size)]

        return self.data[:, jdx], jdx