import sys

import torch, torch.nn as nn

sys.path.insert(0, '.')
from msgwam import config
from msgwam.rays import RayCollection
from msgwam.utils import put

from hyperparameters import network_size
from preprocessing import RAYS_PER_PACKET
from utils import get_batch_pmf

class CoarseNet(nn.Module):
    """
    CoarseNet currently takes in a zonal wind profile and `config.n_source`
    values for each ray volume property, and predicts the properties of one ray
    volume that is intended to replace it. At present, the neural network only
    operates on the zonal wind and the (dr, m, dm) values and returns those,
    with the remaining properties taken from the input or calculated to
    conserve momentum flux.
    """

    props = ['dr', 'k', 'm', 'dm']
    idx = [RayCollection.indices[prop] for prop in props]

    def __init__(self) -> None:
        """Initialize a CoarseNet."""

        super().__init__()

        n_z = config.n_grid - 1
        n_inputs = n_z + 9 * RAYS_PER_PACKET
        n_outputs = len(self.props)

        sizes = [n_inputs] + [512] * network_size
        sizes = sizes + [256, 128, 64, 32, n_outputs]
        args = [nn.BatchNorm1d(n_inputs)]

        for n_in, n_out in zip(sizes[:-1], sizes[1:]):
            args.append(nn.Linear(n_in, n_out))
            args.append(nn.ReLU())

        args = args[:-1] + [nn.Softplus()]
        self.layers = nn.Sequential(*args)
        self.layers.apply(_xavier_init)
        self.to(torch.double)

    def forward(self, u: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
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
        Z
            Tensor containing the properties of the replacement ray volumes
            whose first dimension ranges over the properties in `self.props` and
            whose second dimension ranges over the packets being replaced.

        """

        shape = (-1, 9 * RAYS_PER_PACKET)
        packets = X.transpose(0, 1).reshape(shape)
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
            Tensor containing zonal wind profiles and ray volume properties,
            structured like a batch in the output of `_sample_wave_packets`.
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

        Y = torch.nanmean(X, dim=-1)
        Y = put(Y, cls.idx, output * Y[cls.idx])
        factor = get_batch_pmf(X) / get_batch_pmf(Y)
        Y = put(Y, -1, Y[-1] * factor)

        return Y

def _xavier_init(layer: nn.Module):
    """
    Apply Xavier initialization to a layer if it is an `nn.Linear`.

    Parameters
    ----------
    layer
        Module to potentially initialize.

    """

    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight)