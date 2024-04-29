from abc import ABC, abstractmethod

import torch

from . import config, spectra
from .dispersion import cg_r

class Source(ABC):
    def __init__(self) -> None:
        """
        Initialize the source with data from the spectrum type specified in the
        config file. That parameter must correspond to the name of a function
        defined in spectra.py.

        """

        spectrum_func = getattr(spectra, config.spectrum_type)
        self.data: torch.Tensor = spectrum_func()
        self.n_slots = self.data.shape[1]

    @abstractmethod
    def launch(
        self,
        j: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Each spectrum is discretized into a fixed number of slots, each of which
        corresponds to a column in `self.data`. This function takes in a tensor
        containing the slots of rays that have cleared the launch level and then
        returns the properties of the ray volumes that should be launched.

        For a constant source, this might be as simple as returning the columns
        of `self.data` indexed by `j`. However, a stochastic source might return
        only a subset of the slots indexed in `j`, while a neural network source
        might return a small number of ray volumes, each of which stands in for
        several of the ones requested.

        In addition to the properties of the ray volumes to launch, the source
        returns two helper tensors. The first is a tensor of the slots that are
        replaced by one of the returned ray volumes, and the second is the
        number of slots each returned ray volume is intended to replace.
        
        Parameters
        ----------
        j
            Tensor of slots of ray volumes that have cleared the launch level.

        Returns
        -------
        torch.Tensor
            Tensor of properties of the ray volumes that should be launched.
        torch.Tensor
            Tensor of slots that are replaced by one of the rays in the first
            output tensor.
        torch.Tensor
            Tensor whose length is the number of columns in the first output
            tensor, and whose values indicate how many of the slots in the 
            second output tensor each returned ray volume replaces.
            
        """
        ...

class ConstantSource(Source):
    def launch(
        self,
        j: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Simply return the columns of `self.data` indexed by j."""

        return self.data[:, j], j, torch.ones(len(j)).int()
    
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

    def launch(
        self,
        j: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return `config.epsilon` times the number of rays requested, sampled such
        that faster ray volumes are launched more frequently.
        """

        size = int(torch.floor(self.launch_rate))
        if torch.rand(1) < self.launch_rate - size:
            size = size + 1

        weights = self.cg_source[j]
        weights = weights / weights.sum()
        j = j[torch.multinomial(weights, num_samples=size)]

        return self.data[:, j], j, torch.ones(len(j)).int()