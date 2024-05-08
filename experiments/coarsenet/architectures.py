import sys

import torch, torch.nn as nn

sys.path.insert(0, '.')
from msgwam.rays import RayCollection

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

    props = ['dr', 'm', 'dm']
    idx = [RayCollection.indices[prop] for prop in props]

    def __init__(self) -> None:
        """Initialize a CoarseNet."""

        super().__init__()

        n_inputs = 9 * RAYS_PER_PACKET
        n_outputs = len(self.props)

        self.layers = nn.Sequential(
            nn.BatchNorm1d(n_inputs),
            nn.Linear(n_inputs, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, n_outputs), nn.Softplus()
        )

        self.to(torch.double)
        self.layers.apply(_xavier_init)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Apply the CoarseNet, including unpacking the ray volume data, applying
        the neural network, and repacking the calculated wave properties as
        a spectrum tensor.

        Parameters
        ----------
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
        output = self.layers(torch.nan_to_num(packets))
        means = torch.nanmean(X[self.idx], dim=2)

        return output.T * means
    
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
        Y[cls.idx] = output.detach()
        Y[-1] = Y[-1] * get_batch_pmf(X) / get_batch_pmf(Y)

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