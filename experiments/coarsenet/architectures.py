import sys
sys.path.insert(0, '.')

import torch, torch.nn as nn

from msgwam import config
from msgwam.rays import RayCollection
from msgwam.sources import CoarseSource
from msgwam.utils import put

from hyperparameters import conservative, network_size, root
from utils import (
    root_transform, 
    xavier_init
)

class CoarseNet(nn.Module):
    """
    CoarseNet accepts a zonal wind profile along with a (coarsened) ray volume,
    and predicts scaling factors for that ray volume's properties intended to
    make its online behavior more like that of its constituent rays.
    """

    props_in = ['k', 'm', 'dm', 'dens']
    props_out = ['k', 'm', 'dm', 'dens']

    idx_in = [RayCollection.indices[prop] for prop in props_in]
    idx_out = [RayCollection.indices[prop] for prop in props_out]

    def __init__(self, u_tr: torch.Tensor, Y_tr: torch.Tensor) -> None:
        """
        Initialize a `CoarseNet`, calculating input statistics to be used for
        standardization at prediction time.

        Parameters
        ----------
        u_tr
            Zonal wind profiles in the training set.
        Y_tr
            Coarse ray volumes in the training set.

        """

        super().__init__()
        self._init_stats(u_tr, Y_tr)

        n_z = config.n_grid - 1
        wind_sizes = [1] + [16] * network_size + [1]
        rays_sizes = [len(self.props_in)] + [256] * network_size

        seq_out = n_z - 2 * len(wind_sizes) + 2
        rays_sizes = rays_sizes + [seq_out]
        shared_sizes = [seq_out] + [512] * network_size + [len(self.props_out)]

        wind_args = []
        for a, b in zip(wind_sizes[:-1], wind_sizes[1:]):
            wind_args.append(nn.Conv1d(a, b, 3))
            wind_args.append(nn.ReLU())

        rays_args = []
        for a, b in zip(rays_sizes[:-1], rays_sizes[1:]):
            rays_args.append(nn.Linear(a, b, 3))
            rays_args.append(nn.ReLU())

        shared_args = []
        for a, b in zip(shared_sizes[:-1], shared_sizes[1:]):
            shared_args.append(nn.Linear(a, b, 3))
            shared_args.append(nn.ReLU())

        shared_args = shared_args[:-1] + [nn.Softplus()]

        self.wind_layers = nn.Sequential(*wind_args)
        self.rays_layers = nn.Sequential(*rays_args)
        self.shared_layers = nn.Sequential(*shared_args)

        self.wind_layers.apply(xavier_init)
        self.rays_layers.apply(xavier_init)
        self.shared_layers.apply(xavier_init)
        self.to(torch.double)

    def forward(self, u: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Apply the `CoarseNet` to a zonal wind profile and tensor of coarse ray
        volume properties.

        Parameters
        ----------
        u
            Tensor containing a single zonal wind profile.
        Y
            Tensor containing a batch of coarse ray volume properties.

        Returns
        -------
        torch.Tensor
            Tensor of positive scale factors whose first dimension ranges over
            the properties in `props_out` and whose second dimension ranges over
            the ray volumes being replaced.

        """

        u = u[:, None].expand(-1, Y.shape[1])
        Y = root_transform(Y, root)[self.idx_in]
        u = self._standardize(u.T, mode='u')
        Y = self._standardize(Y.T, mode='Y')
        
        p = self.wind_layers(u[:, None])[:, 0]
        q = self.rays_layers(Y)

        return self.shared_layers(p + q).T

    def _init_stats(self, u: torch.Tensor, Y: torch.Tensor) -> None:
        """
        Calculate the means and standard deviations to be used to standardize
        the inputs to `forward`.

        Parameters
        ----------
        u
            Batches of zonal wind profiles.
        Y
            Batches of coarse ray volumes.

        """

        Y = root_transform(Y, root)
        Y = Y[:, self.idx_in].transpose(1, 2).flatten(0, 1)

        self.Y_means = torch.mean(Y, dim=0)
        self.Y_stds = torch.std(Y, dim=0)
        self.Y_sdx = self.Y_stds > 0

        self.u_means = torch.mean(u, dim=0)
        self.u_stds = torch.std(u, dim=0)
        self.u_sdx = self.u_stds > 0
        
    def _standardize(self, a: torch.Tensor, mode: str) -> torch.Tensor:
        """
        Standardize a tensor of either zonal wind profiles or coarse ray volume
        properties to have zero mean and unit variance.

        Parameters
        ----------
        a
            Tensor to standardize
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
    
    @classmethod
    def build_spectrum(
        cls,
        Y: torch.Tensor,
        output: torch.Tensor
    ) -> torch.Tensor:
        """
        Create a complete source spectrum, using the outputs of this model to
        scale the appropriate ray volume properties and ensuring that the result
        has the same momentum flux as the original.

        Parameters
        ----------
        Y
            Batch of coarse ray volumes to be replaced.
        output
            Tensor of scale factors for the properties in `props_out`, obtained
            by calling this model on `Y` and a zonal wind profile.

        """

        fluxes = CoarseSource._get_fluxes(Y)
        Y = put(Y, cls.idx_out, output * Y[cls.idx_out])

        if conservative:
            factor = fluxes / CoarseSource._get_fluxes(Y)
            Y = put(Y, 8, Y[8] * factor)

        return Y
