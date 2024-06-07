import sys
sys.path.insert(0, '.')

import torch, torch.nn as nn

from msgwam import config
from msgwam.rays import RayCollection
from msgwam.sources import CoarseSource
from msgwam.utils import put

from hyperparameters import network_size, root
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
    props_out = ['dr', 'k', 'm', 'dm']

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
        n_in = n_z + len(self.props_in)
        n_out = len(self.props_out)

        sizes = [n_in] + [512] * network_size
        sizes = sizes + [256, 128, 64, 32, n_out]
        args = []

        for a, b in zip(sizes[:-1], sizes[1:]):
            args.append(nn.Linear(a, b))
            args.append(nn.ReLU())

        args = args[:-1] + [nn.Softplus()]
        self.layers = nn.Sequential(*args)
        self.layers.apply(xavier_init)
        self.to(torch.double)

    def forward(self, u: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Apply the `CoarseNet`, including unpacking the ray volume data, stacking
        it with the zonal wind data, and passing it through the neural network.

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

        Y = root_transform(Y, root)[self.idx_in]
        u = u[:, None].expand(-1, Y.shape[1])
        stacked = torch.hstack((u.T, Y.T))

        return self.layers(self._standardize(stacked)).T

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
        shape = (-1, -1, Y.shape[-1])
        u = u[..., None].expand(*shape).transpose(1, 2).flatten(0, 1)
        Y = Y[:, self.idx_in].transpose(1, 2).flatten(0, 1)
        
        stacked = torch.hstack((u, Y))
        self.means = torch.mean(stacked, dim=0)
        self.stds = torch.std(stacked, dim=0)
        self.sdx = self.stds != 0

    def _standardize(self, stacked: torch.Tensor) -> torch.Tensor:
        """
        Standardize neural network inputs using means and standard deviations
        calculated at initialization.

        Parameters
        ----------
        stacked
            Tensor of inputs to the neural network, consisting of zonal wind and
            coarse wave packet information side by side.

        Returns
        -------
        torch.Tensor
            Standardized data.

        """

        out = torch.zeros_like(stacked)
        out[:, self.sdx] = (
            (stacked - self.means)[:, self.sdx] /
            self.stds[self.sdx]
        )

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
        factor = fluxes / CoarseSource._get_fluxes(Y)
        Y = put(Y, 8, Y[8] * factor)

        return Y
