import sys

import torch, torch.nn as nn

sys.path.insert(0, '.')
from msgwam import config
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
    dprops = torch.tensor([10, 2 * torch.pi / 50, 2 * torch.pi / 50])[:, None]
    idx = [RayCollection.indices[prop] for prop in props]

    def __init__(self) -> None:
        """Initialize a CoarseNet."""

        super().__init__()

        n_wind = config.n_grid - 1
        n_inputs = n_wind + len(self.props) * RAYS_PER_PACKET
        n_outputs = len(self.props)

        self.layers = nn.Sequential(
            nn.Linear(n_inputs, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, n_outputs), nn.Softplus()
        )

        self.to(torch.double)
        # self.layers.apply(_xavier_init)

    def forward(self, wind: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """
        Apply the CoarseNet, including unpacking the ray volume data, applying
        the neural network, and repacking the calculated wave properties as
        a spectrum tensor.

        Parameters
        ----------
        wind
            Tensor containing the fixed zonal and meridional wind profiles to
            use while integrating each packet.
        X
            Tensor containing zonal wind profiles and ray volume properties,
            structured like a batch in the output of `_sample_wave_packets`.
        
        Returns
        -------
        Z
            Tensor containing the properties of the replacement ray volumes
            whose first row ranges over the properties in `self.props` and whose
            second row ranges over the packets being replaced.

        """

        wind = wind[0].expand(X.shape[1], -1)
        shape = (-1, len(self.props) * RAYS_PER_PACKET)
        data = X[self.idx].transpose(0, 1).reshape(shape)

        stacked = torch.hstack((wind, torch.nan_to_num(data)))
        output = self.layers(stacked)

        means = X[self.idx]
        means[means == 0] = torch.nan
        means = torch.nanmean(means, dim=2)

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