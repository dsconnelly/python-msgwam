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
        self._init_stats(Y_tr)

        n_z = config.n_grid - 1
        wind_sizes = [1] + [16] * network_size + [1]
        rays_sizes = [len(PROPS_IN)] + [256] * network_size

        seq_out = n_z - 2 * len(wind_sizes) + 2
        rays_sizes = rays_sizes + [seq_out]
        shared_sizes = [seq_out] + [512] * network_size + [n_z]

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
        
        """

        u, X = self._split(Y)
        u = standardize(u, self.u_means, self.u_stds)
        X = standardize(X, self.X_means, self.X_stds)
        u = u.view(u.shape[0], 1, u.shape[1])

        p = self.wind_layers(u)[:, 0]
        q = self.rays_layers(X)

        return self.shared_layers(p + q)
    
    def get_extra_state(self) -> dict[str, torch.Tensor]:
        """
        
        """

        return {
            'u_means' : self.u_means,
            'X_means' : self.X_means,
            'u_stds' : self.u_stds,
            'X_stds' : self.X_stds
        }
    
    def set_extra_state(self, state: dict[str, torch.Tensor]) -> None:
        """
        
        """
        
        self.u_means = state['u_means']
        self.X_means = state['X_means']
        self.u_stds = state['u_stds']
        self.X_stds = state['X_stds']

    def _split(self, Y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        
        """

        n_z = config.n_grid - 1
        u, X = Y[:, :n_z], Y[:, n_z:]
        X = root_transform(X, root)

        return u, X

    def _init_stats(self, Y: torch.Tensor) -> torch.Tensor:
        """
        
        """

        u, X = self._split(Y)

        self.u_means = u.mean(dim=0)
        self.X_means = X.mean(dim=0)
        
        self.u_stds = u.std(dim=0)
        self.X_stds = X.std(dim=0)

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
