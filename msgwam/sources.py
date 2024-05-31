from abc import ABC, abstractmethod

import torch

from . import config
from .dispersion import cg_r, cp_x
from .spectra import get_spectrum

class Source(ABC):
    def __init__(self) -> None:
        """
        Initialize the source with data from the spectrum type specified in the
        config file. That parameter must correspond to the name of a function
        defined in spectra.py.

        """

        self.data = get_spectrum()
        self.n_slots = self.data.shape[1]

    @abstractmethod
    def launch(
        self,
        j: torch.Tensor,
        u: torch.Tensor
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
        u
            Current zonal wind profile (can be discarded by subclasses).

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
        j: torch.Tensor,
        _
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Simply return the columns of `self.data` indexed by j."""

        return self.data[:, j], j, torch.ones(len(j)).int()
    
class NetworkSource(Source):
    def __init__(self) -> None:
        """
        At initialization, the neural network source divides the source up into
        packets that will be replaced later.
        """

        super().__init__()
        idx = torch.argsort(cp_x(*self.data[2:5]))
        self.data = self.data[:, idx]

        speedup = config.rays_per_packet
        self.dt_launch = speedup * config.dt
        self.repeats = self.dt_launch * cg_r(*self.data[2:5]) / self.data[1]

        root = int(speedup ** 0.5)
        columns = torch.arange(self.n_slots)
        self.sdx = torch.floor_divide(columns, root)
        
        self.model = torch.jit.load(config.model_path)

    def launch(
        self,
        j: torch.Tensor,
        u: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Takes the requested slots, calculates how many times each ray volume
        would have cleared the lower boundary since the last launch, then builds
        packets of nearby ray volumes and their repeats and coarsens them using
        the loaded neural network. 
        """

        repeats = torch.floor(self.repeats[j]).int()
        idx = torch.rand(len(j)) < self.repeats[j] - repeats
        repeats[idx | (repeats == 0)] += 1

        copies = torch.repeat_interleave(j, repeats)
        copies = copies[torch.randperm(repeats.sum())]
        copies = copies[torch.argsort(self.sdx[copies])]

        packet_ids = self._get_packet_ids(copies)
        n_packets = packet_ids.max()

        X = torch.nan * torch.zeros((9, n_packets, config.rays_per_packet))
        counts = torch.zeros(n_packets).int()
        max_seen = -1

        for k in range(n_packets):
            pdx = packet_ids == k
            X[:, k, :pdx.sum()] = self.data[:, copies[pdx]]
            counts[k] = len(torch.unique(copies[pdx & (copies > max_seen)]))
            max_seen = max(max_seen, copies[pdx].max())

        with torch.no_grad():
            spectrum = self.model(u, X)

        return spectrum, j, counts

    def _get_packet_ids(self, j: torch.Tensor) -> torch.Tensor:
        """
        Given a tensor of slots of ray volumes that need to be launched, group
        them into packets of size no more than `config.rays_per_packet`.
        
        Parameters
        ----------
        j
            Tensor of slots of ray volumes to coarsen, including repeats.

        Returns
        -------
        torch.Tensor
            Tensor indicating, for each ray volume in `j`, which packet it
            should belong to.

        """

        _, counts = torch.unique(self.sdx[j], return_counts=True)
        fulls = torch.floor_divide(counts, config.rays_per_packet)
        rems = torch.remainder(counts, config.rays_per_packet).int()
        totals = fulls + (rems > 0).int()

        n_packets = totals.sum()
        packet_ids = torch.arange(n_packets)
        sizes = config.rays_per_packet * torch.ones(n_packets).int()

        lasts = (torch.cumsum(totals, 0) - 1)[rems > 0]
        sizes[lasts] = rems[rems > 0]

        return torch.repeat_interleave(packet_ids, sizes.int())

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
        j: torch.Tensor,
        _
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