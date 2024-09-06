from abc import ABC, abstractmethod

import torch, torch.nn as nn

from msgwam import config
from msgwam.constants import PROP_NAMES

from hyperparameters import network_size, root
from utils import root_transform

class StandardizerNet(nn.Module, ABC):
    """
    `StandardizerNet` provides shared functions for neural network architectures
    that accept wind profiles coarse ray volume properties as input, since the
    preprocessing operations for those inputs are somewhat specific.    
    """

    idx_in: list[int]

    def __init__(self, u_tr: torch.Tensor, Y_tr: torch.Tensor) -> None:
        """
        At initialization, a `StandardizerNet` takes the training zonal wind and
        coarse ray volume data and stores the necessary statistics.

        Parameters
        ----------
        u_tr
            Zonal wind profiles in the training set.
        Y_tr
            Coarse ray volumes in the training set.

        """

        super().__init__()
        self._init_stats(u_tr, Y_tr)

    def forward(self, u: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        `StandardizerNet` handles the input preprocessing, and then calls the
        `_predict` function that subclasses are expected to implement.

        Parameters
        ----------
        u
            Tensor containing a single zonal wind profile.
        Y
            Tensor containing a batch of coarse ray volume properties.

        """

        u = u[:, None].expand(-1, Y.shape[1])
        Y = root_transform(Y, root)[self.idx_in]
        u = self._standardize(u.T, mode='u')
        Y = self._standardize(Y.T, mode='Y')

        return self._predict(u, Y)

    def _init_stats(self, u: torch.Tensor, Y: torch.Tensor) -> None:
        """
        Calculate the means and standard deviations to be used to standardize
        the inputs to `forward`, making sure to apply the appropriate root
        transformation to the ray volume properties.

        Parameters
        ----------
        u
            Batches of zonal wind profiles.
        Y
            Batches of coarse ray volumes.
        """

        self.u_means = torch.mean(u, dim=0)
        self.u_stds = torch.std(u, dim=0)
        self.u_sdx = self.u_stds > 0

        Y = root_transform(Y, root)
        Y = Y[:, self.idx_in].transpose(1, 2).flatten(0, 1)

        self.Y_means = torch.mean(Y, dim=0)
        self.Y_stds = torch.std(Y, dim=0)
        self.Y_sdx = self.Y_stds > 0

    @abstractmethod
    def _predict(self, u: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Apply the network to standardized inputs.

        Parameters
        ----------
        u
            Single zonal wind profile, standardized and broadcast.
        Y
            Standardized batch of coarse ray volumes.

        Returns
        -------
        torch.Tensor
            Neural network output specific to each subclass.

        """
        ...

    def _standardize(self, a: torch.Tensor, mode: str) -> torch.Tensor:
        """
        Standardize a tensor of either zonal wind profiles or coarse ray volume
        properties to have zero mean and unit variance.

        Parameters
        ----------
        a
            Tensor to standardize.
        mode
            Whether to use statistics for `u` or `Y`.

        Returns
        -------
        torch.Tensor
            Standardized tensor.

        """

        means = getattr(self, f'{mode}_means')
        stds = getattr(self, f'{mode}_stds')
        sdx = getattr(self, f'{mode}_sdx')

        out = torch.zeros_like(a)
        out[:, sdx] = (a - means)[:, sdx] / stds[sdx]

        return out

class EmulatorNet(StandardizerNet):
    """
    `EmulatorNet` accepts a zonal wind profile along with a (coarsened) ray
    volume and predicts the time-mean momentum flux profile associated with that
    ray volume over the integration period.
    """

    props_in = ['k', 'm', 'dm', 'dens']
    idx_in = [PROP_NAMES.index(prop) for prop in props_in]

    def __init__(self, u_tr: torch.Tensor, Y_tr: torch.Tensor) -> None:
        """
        
        """

        super().__init__(u_tr, Y_tr)

        n_z = config.n_grid - 1
        wind_sizes = [1] + [16] * network_size + [1]

        wind_args = []
        for a, b in zip(wind_sizes[:-1], wind_sizes[1:]):
            wind_args.append(nn.Conv1d(a, b, 3))
            wind_args.append(nn.ReLU())

        rays_sizes = [len(self.props_in)] + [256] * network_size
        rays_sizes = rays_sizes + [n_z - 2 * len(wind_sizes) + 2]

        rays_args = []
        for a, b in zip(rays_sizes[:-1], rays_sizes[1:]):
            rays_args.append(nn.Linear(a, b))
            rays_args.append(nn.ReLU())

        shared_sizes = [512] * network_size + [n_z]
        shared_sizes = [rays_sizes[-1]] + shared_sizes

        shared_args = []
        for a, b in zip(shared_sizes[:-1], shared_sizes[1:]):
            shared_args.append(nn.Linear(a, b))
            shared_args.append(nn.ReLU())

        self.wind_layers = nn.Sequential(*wind_args)
        self.rays_layers = nn.Sequential(*rays_args)
        self.shared_layers = nn.Sequential(*shared_args)

    def _predict(self, u: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        
        """

        p = self.wind_layers(u[:, None]).squeeze()
        q = self.rays_layers(Y)

        return self.shared_layers(p + q)