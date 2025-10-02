import torch, torch.nn as nn

from msgwam import config

from ...hyperparameters import architectures as hp

from .utils import apply_blocks, get_block, xavier_init

class BulkNet(nn.Module):
    def __init__(self) -> None:
        """
        At initialization, a `BulkNet` creates a series of blocks that will be
        used with skip connections at prediction time.
        """

        super().__init__()

        self._blocks = self._init_blocks()
        self.apply(xavier_init)
        self.to(torch.double)

    def forward(self,
        wind: torch.Tensor,
        M: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply the forward model.

        Parameters
        ----------
        M
            Tensor of bulk momentum profiles.
        cg
            Tensor of bulk group velocity profiles.
        wind
            Tensor of mean wind profiles, negated as necessary.

        Returns
        -------
        torch.Tensor, torch.Tensor
            Updated bulk momentum and group velocity profiles, respectively.

        """

        X = torch.hstack((wind, M))
        out = apply_blocks(self._blocks, X)
        out = out - out.mean(dim=1)[:, None]

        return out
        
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
            sizes = [self._n_inputs] + [hp.n_hidden] * hp.block_depth
            sizes = sizes + [self._n_outputs if final else self._n_inputs]
            blocks.append(get_block(sizes, final))

        return blocks
    
    @property
    def _n_inputs(self) -> int:
        """
        A `Bulknet` accepts a mean wind profile, a buoyancy frequency profile,
        and one bulk momentum profile for each phase speed bin. Each profile has
        `config.n_grid - 1` values.
        """

        return (config.n_grid - 1) * (2 + hp.n_bins)

    @property
    def _n_outputs(self) -> int:
        """
        For each wavenumber quadrant, a `BulkNet` predicts one momentum profile
        for each phase speed bin, as well as a prediction of the dissipative
        momentum loss at each level.
        """

        return (config.n_grid - 1) * (1 + hp.n_bins)
