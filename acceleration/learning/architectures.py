import numpy as np
import torch, torch.nn as nn

from msgwam import config
from msgwam.utils import get_vertical_grids

from ..hyperparameters import architectures as hp

from .utils import apply_blocks, xavier_init

class BulkNet(nn.Module):
    def __init__(self) -> None:
        """
        At initialization, a `BulkNet` sets up normalization layers for the
        inputs and instantiates the main neural network layers.
        """

        super().__init__()
        self._cg_norm = nn.BatchNorm1d(config.n_grid, affine=False)
        self._wind_norm = nn.BatchNorm1d(config.n_grid - 1, affine=False)
        self._blocks = self._init_blocks()

        z, _ = get_vertical_grids()
        dz = np.diff(z)[0] * np.ones_like(z)
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
        Y = apply_blocks(self._blocks, X)

        M = Y[:, :config.n_grid + 1]
        cg = Y[:, config.n_grid + 1:]
        M = M / M.sum(dim=1)[:, None]

        return M[:, :-1] * budget / self._dz, cg
    
    def _get_block(self, final: bool) -> nn.Sequential:
        """
        Build a block of fully-connected layers for the neural network, placing
        batch normalization layers according to hyperparameter settings.

        Parameters
        ----------
        final
            Whether this is the last block in the network. If so, the number of
            outputs will be set accordingly and the last layer will be a `ReLU`.
        
        Returns
        -------
        nn.Sequential
            Module containing the resulting layers.

        """

        sizes = [self._n_inputs] + [hp.n_hidden] * hp.block_depth
        sizes = sizes + [self._n_outputs if final else self._n_inputs]
        
        args = []
        for a, b in zip(sizes[:-1], sizes[1:]):
            args = args + [nn.Linear(a, b), nn.ReLU()]

            if hp.batch_norm_pos != 0:
                k = len(args) - (hp.batch_norm_pos == -1)
                args.insert(k, nn.BatchNorm1d(b))

        if final:
            while not isinstance(args[-1], nn.ReLU):
                args = args[:-1]

        return nn.Sequential(*args)

    def _init_blocks(self) -> nn.ModuleList:
        """
        Initialize the main neural network layers.

        Returns
        -------
        nn.ModuleList
            List of blocks to apply at prediction time.

        """

        blocks = nn.ModuleList()
        for i in range(hp.n_blocks):
            final = i == hp.n_blocks - 1
            blocks.append(self._get_block(final))

        return blocks

    @property
    def _n_inputs(self) -> int:
        """
        At present, the `BulkNet` simply takes in the bulk momentum and group
        velocity profiles from the previous time step, the appropriate component
        of the mean wind, and the added source momentum.
        """

        return 3 * config.n_grid

    @property
    def _n_outputs(self) -> int:
        """
        For each wavenumber quadrant, a `BulkNet` predicts two profiles, one for
        bulk momentum and the other for bulk group velocity. Each as a value for
        each vertical grid face, and the former has one extra output
        corresponding to unused momentum.
        """

        return 2 * config.n_grid + 1

