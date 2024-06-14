import torch, torch.nn as nn

from msgwam import config

from hyperparameters import network_size, root
from preprocessing import PROPS_IN
from utils import root_transform, standardize

class PurgeNet(nn.Module):
    def __init__(self, Y_tr: torch.Tensor) -> None:
        """
        Initialize a `PurgeNet`, calculating input statistics to be used for
        standardization at prediction time.

        Parameters
        ----------
        Y_tr
            Tensor of training inputs from which to calculate statistics.

        """

        super().__init__()

        Y_tr = self._transform(Y_tr)
        self.means = Y_tr.mean(dim=0)
        self.stds = Y_tr.std(dim=0)

        n_z = config.n_grid - 1
        wind_sizes = [1, 16, 32] + [64] * network_size + [1]
        rays_sizes = [len(PROPS_IN)] + [256] * network_size

        n_shared = n_z - 2 * len(wind_sizes) + 2 + rays_sizes[-1]
        shared_sizes = [n_shared] + [512] * (network_size + 1) + [n_z]

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

        self.wind_layers = nn.Sequential(*wind_args)
        self.rays_layers = nn.Sequential(*rays_args)
        self.shared_layers = nn.Sequential(*shared_args[:-1])

        self.wind_layers.apply(_xavier_init)
        self.rays_layers.apply(_xavier_init)
        self.shared_layers.apply(_xavier_init)
        self.to(torch.double)

    def forward(self, Y: torch.Tensor) -> torch.Tensor:
        """
        Standardize the inputs and apply the `PurgeNet`.

        Parameters
        ----------
        Y
            Tensor of neural network inputs.

        Returns
        -------
        torch.Tensor
            Tensor of neural network outputs estimate (possibly standardized)
            mean momentum flux profiles for each input ray volume.

        """

        n_z = config.n_grid - 1
        Y = standardize(self._transform(Y), self.means, self.stds)
        u, X = Y[:, None, :n_z], Y[:, n_z:]

        p = self.wind_layers(u).squeeze()
        q = self.rays_layers(X)

        return self.shared_layers(torch.hstack((p, q)))

    def _transform(self, Y: torch.Tensor):
        """
        Transform a neural network input using a signed root transform.

        Parameters
        ----------
        Y
            Tensor of neural network inputs.

        Returns
        -------
        torch.Tensor
            Transformed data.

        """

        n_z = config.n_grid - 1
        data = root_transform(Y[:, n_z:], root)
        return torch.hstack((Y[:, :n_z], data))

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
