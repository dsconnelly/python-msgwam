import numpy as np
import torch, torch.nn as nn

from msgwam import config
from msgwam.utils import get_vertical_grids

from .utils import xavier_init

class SupervolumeNet(nn.Module):
    def __init__(self) -> None:
        """
        At initialization, a `SupervolumeNet` sets up normalization layers for
        the inputs and instantiates the main neural network layers.
        """

        super().__init__()
        self._cg_norm = nn.BatchNorm1d(config.n_grid, affine=False)
        self._wind_norm = nn.BatchNorm1d(config.n_grid - 1, affine=False)
        self._layers = self._init_layers()

        z, _ = get_vertical_grids()
        dz = np.diff(z)[0] * np.ones_like(z)
        dz[0] = dz[-1] = 0.5 * z[1]
        self._dz = torch.as_tensor(dz)

        self.apply(xavier_init)
        self.to(torch.double)

    def forward(
        self,
        M: torch.Tensor,
        cg: torch.Tensor,
        source: torch.Tensor,
        wind: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply the forward model.

        Parameters
        ----------
        M
            Tensor of bulk momentum profiles.
        cg
            Tensor of bulk group velocity profiles.
        source
            Tensor of source momenta added at the current time step.
        wind
            Tensor of mean wind profiles, negated as necessary.
            
        Returns
        -------
        torch.Tensor, torch.Tensor
            Updated bulk momentum and group velocity profiles, respectively, for
            each sample. The bulk momentum profiles are returned normalized by
            the budget at the next time step - that is, the total input momentum
            plus that added by the source.

        """

        X = torch.hstack((M * self._dz, source))
        budget = X.sum(dim=1)[:, None]
        X = X / budget

        cg = self._cg_norm(cg)
        wind = self._wind_norm(wind)
        X = torch.hstack((wind, X, cg))

        Y = self._layers(X)
        M = Y[:, :config.n_grid + 1]
        cg = Y[:, config.n_grid + 1:]
        M = M / M.sum(dim=1)[:, None]

        return M[:, :-1] * budget / self._dz, cg
    
    def _init_layers(self) -> nn.Sequential:
        """
        Initialize the main neural network layers.

        Returns
        -------
        nn.Sequential
            Sequential instance containing the linear and activation layers.

        """

        sizes = [self._n_inputs] + [256] * 5 + [self._n_outputs]
        
        args = []
        for a, b in zip(sizes[:-1], sizes[1:]):
            args.extend([
                nn.Linear(a, b),
                nn.ReLU()
            ])

        return nn.Sequential(*args)

    @property
    def _n_inputs(self) -> int:
        """
        At present, the `SupervolumeNet` simply takes in the bulk momentum and
        group velocity profiles from the previous time step, the appropriate
        component of the mean wind, and the added source momentum.
        """

        return 3 * config.n_grid

    @property
    def _n_outputs(self) -> int:
        """
        For each wavenumber quadrant, a `SupervolumeNet` predicts two profiles,
        one for bulk momentum and the other for bulk group velocity. Each as a
        value for each vertical grid face, and the former has one extra output
        corresponding to unused momentum.
        """

        return 2 * config.n_grid + 1

