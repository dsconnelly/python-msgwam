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
        At initialization, the neural network source calculates how many times
        each source ray volume would clear the bottom boundary between launches.
        """

        super().__init__()

        self.width = int(config.rays_per_packet ** 0.5)
        distances = config.dt_launch * cg_r(*self.data[2:5])
        self.n_repeats = distances / self.data[1]

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

        j, _ = torch.sort(j)
        n_repeats = torch.floor(self.n_repeats[j]).int()
        idx = torch.rand(len(j)) < self.n_repeats[j] - n_repeats
        n_repeats[idx | (n_repeats == 0)] += 1

        copies = torch.repeat_interleave(j, n_repeats)
        data = self._stack(self.data[:, j], n_repeats)
        packet_ids = self._get_packet_ids(copies, self.width)
        n_packets = packet_ids.max()

        X = torch.nan * torch.zeros((9, n_packets, config.rays_per_packet))
        counts = torch.zeros(n_packets).int()
        max_seen = -1

        for k in range(n_packets):
            pdx = packet_ids == k
            X[:, k, :pdx.sum()] = data[:, pdx]
            j_new = copies[pdx & (copies > max_seen)]
            counts[k] = len(torch.unique(j_new))

            if len(j_new) > 0:
                max_seen = max(max_seen, j_new.max())

        with torch.no_grad():
            spectrum = self.model(u, X)

        return spectrum, j, counts

    @staticmethod
    def _get_packet_ids(j: torch.Tensor, width: int) -> torch.Tensor:
        """
        Given a tensor of slots in the source spectrum that need to be launched
        (possibly including repeats), group them into packets, each of which
        only includes ray volumes within `self.width` of each other and has no
        more than `config.rays_per_packet` members.

        Parameters
        ----------
        j
            Tensor of slots of ray volumes to group.
        width
            How many slots in the source spectrum each packet may span.

        Returns
        -------
        torch.Tensor
            Tensor indicating which packet each ray volume in `j` belongs to.

        """

        idx = torch.argsort(j)
        rdx = torch.argsort(idx)
        j = j[idx]

        channels = torch.floor_divide(j, width)
        _, counts = torch.unique(channels, return_counts=True)

        fulls = torch.floor_divide(counts, config.rays_per_packet)
        rems = torch.remainder(counts, config.rays_per_packet)
        totals = fulls + (rems > 0)
        n_packets = totals.sum()

        packet_ids = torch.arange(n_packets)
        sizes = config.rays_per_packet * torch.ones_like(packet_ids)
        lasts = (torch.cumsum(totals, 0) - 1)[rems > 0]
        sizes[lasts] = rems[rems > 0]

        return torch.repeat_interleave(packet_ids, sizes)[rdx]

    @staticmethod
    def _stack(spectrum: torch.Tensor, n: int | torch.Tensor) -> torch.Tensor:
        """
        Given a tensor of source ray volumes, return a spectrum with multiple
        copies of the spectrum stacked in the vertical dimension.

        Parameters
        ----------
        spectrum
            Tensor containing source ray volumes.
        n
            How many times to stack the spectrum. Can be a tensor specifying a
            repeat count for each column, or a single integer to be used for all
            columns in the spectrum.

        Returns
        -------
        torch.Tensor
            Stacked spectrum.

        """

        if isinstance(n, int):
            n = n * torch.ones(spectrum.shape[1]).int()

        mods = torch.repeat_interleave(n, n)
        spectrum = torch.repeat_interleave(spectrum, n, dim=1)        
        cols = torch.arange(spectrum.shape[1])

        for i in range(n.max()):
            jdx = torch.remainder(cols, mods) == i
            spectrum[0, jdx] = spectrum[0, jdx] - i * spectrum[1, jdx]

        return spectrum

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