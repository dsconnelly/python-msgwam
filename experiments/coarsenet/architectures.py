import sys
sys.path.insert(0, '.')

import torch, torch.nn as nn

from msgwam import config
from msgwam.rays import RayCollection
from msgwam.utils import put

from hyperparameters import network_size
from preprocessing import RAYS_PER_PACKET
from utils import get_batch_pmf, get_base_replacement, xavier_init

class CoarseNet(nn.Module):
    """
    CoarseNet currently takes in a zonal wind profile and `config.n_source`
    values for each ray volume property, and predicts the properties of one ray
    volume that is intended to replace it. At present, the neural network only
    operates on the zonal wind and the (dr, m, dm) values and returns those,
    with the remaining properties taken from the input or calculated to
    conserve momentum flux.
    """

    props_in = ['r', 'k', 'm', 'dm', 'dens']
    props_out = ['dr', 'k', 'm', 'dm', 'dens']

    idx_in = [RayCollection.indices[prop] for prop in props_in]
    idx_out = [RayCollection.indices[prop] for prop in props_out]

    def __init__(self) -> None:
        """Initialize a CoarseNet."""

        super().__init__()

        n_z = config.n_grid - 1
        n_in = n_z + len(self.props_in) * RAYS_PER_PACKET
        n_out = len(self.props_out)

        sizes = [n_in] + [512] * network_size
        sizes = sizes + [256, 128, 64, 32, n_out]
        args = [nn.BatchNorm1d(n_in, track_running_stats=False)]

        for a, b in zip(sizes[:-1], sizes[1:]):
            args.append(nn.Linear(a, b))
            args.append(nn.ReLU())

        args = args[:-1] + [nn.Softplus()]
        self.layers = nn.Sequential(*args)
        self.layers.apply(xavier_init)
        self.to(torch.double)

    def forward(
        self,
        u: torch.Tensor,
        X: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply the CoarseNet, including unpacking the ray volume data, applying
        the neural network, and repacking the calculated wave properties as
        a spectrum tensor.

        Parameters
        ----------
        u
            Tensor containing a single zonal wind profile.
        X
            Tensor containing ray volume properties, structured like a batch in
            the output of `_sample_wave_packets`.

        Returns
        -------
        torch.Tensor
            Tensor containing positive scale factors whose first dimension
            ranges over the properties in `props_out` and whose second dimension
            ranges over the packets being replaced.

        """

        shape = (-1, len(self.props_in) * RAYS_PER_PACKET)
        packets = X[self.idx_in].transpose(0, 1).reshape(shape)

        u = u[None].expand(X.shape[1], -1)
        stacked = torch.hstack((u, packets))
        output = self.layers(torch.nan_to_num(stacked))

        return output.T
        
    @classmethod
    def build_spectrum(
        cls,
        X: torch.Tensor,
        output: torch.Tensor
    ) -> torch.Tensor:
        """
        Assemble a complete source spectrum, using the output of this model for
        ray volume properties in `self.props` and the average of the original
        packet values for the others. 

        Parameters
        ----------
        X
            Tensor containing ray volume properties for each packet, structured
            like a batch in the output of `_sample_wave_packets`.
        output
            Tensor containing the properties of the replacement ray volumes, as
            in the output of `self.forward`.

        Returns
        -------
        torch.Tensor
            Tensor containing source ray volume properties of the proper shape
            to be passed to the integrator. Will be detached from the provided
            `output` tensor.

        """

        Y = get_base_replacement(X)
        Y = put(Y, cls.idx_out, output * Y[cls.idx_out])
        factor = get_batch_pmf(X) / get_batch_pmf(Y)
        Y = put(Y, 8, Y[8] * factor)

        return Y
