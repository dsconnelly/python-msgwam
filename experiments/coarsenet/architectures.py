import sys
sys.path.insert(0, '.')

import torch, torch.nn as nn

from msgwam import config
from msgwam.rays import RayCollection
from msgwam.utils import put

from hyperparameters import network_size, root
from utils import (
    get_batch_pmf,
    get_base_replacement,
    root_transform, 
    xavier_init
)

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

    def __init__(self, u_tr: torch.Tensor, X_tr: torch.Tensor) -> None:
        """
        Initialize a CoarseNet, calculating input means and standard deviations
        to be used for standardization later.
        
        Parameters
        ----------
        u_tr
            Zonal wind profiles in the training set.
        X_tr
            Wave packets in the training set.
        
        """

        super().__init__()
        self._init_stats(u_tr, X_tr)

        n_z = config.n_grid - 1
        n_in = n_z + len(self.props_in) * config.rays_per_packet
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

        X = root_transform(X, root)
        shape = (-1, len(self.props_in) * config.rays_per_packet)
        packets = X[self.idx_in].transpose(0, 1).reshape(shape)

        u = u[None].expand(X.shape[1], -1)
        stacked = torch.nan_to_num(torch.hstack((u, packets)))
        output = self.layers(self._normalize(stacked))

        return output.T
    
    def _init_stats(self, u: torch.Tensor, X: torch.Tensor) -> None:
        """
        Calculate the means and standard deviations to be used to normalize the
        inputs to `forward`.

        Parameters
        ----------
        u
            Batches of zonal wind profiles.
        X
            Batches of wave packets.

        """    
        
        u_means = u.mean(dim=0)
        u_stds = u.std(dim=0)

        X = root_transform(X[:, self.idx_in], root)
        prop_means = torch.nanmean(X, dim=(0, 2, 3))

        errors = (X - prop_means[..., None, None]) ** 2
        prop_vars = torch.nanmean(errors, dim=(0, 2, 3))
        prop_stds = torch.sqrt(prop_vars)

        X_means = torch.repeat_interleave(prop_means, config.rays_per_packet)
        X_stds = torch.repeat_interleave(prop_stds, config.rays_per_packet)

        self.means = torch.cat([u_means, X_means])
        self.stds = torch.cat([u_stds, X_stds])
        self.sdx = self.stds != 0

    def _normalize(self, stacked: torch.Tensor) -> torch.Tensor:
        """
        Normalize neural network inputs using the means and standard deviations
        calculated at initialization.

        Parameters
        ----------
        stacked
            Tensor of inputs to the neural network, consisting of zonal wind and
            wave packet information side by side.

        Returns
        -------
        torch.Tensor
            Normalized data.

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
