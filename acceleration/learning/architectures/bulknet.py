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

    def forward(
        self,
        windN: torch.Tensor,
        M: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply the forward model.

        Parameters
        ----------
        windN
            Tensor of wind, buoyancy frequency, and latitude data.
        M
            Tensor of bulk momentum profiles.

        Returns
        -------
        torch.Tensor
            Updated bulk momentum profiles in each bin.

        """

        X = torch.hstack((windN, M))
        out = apply_blocks(self._blocks, X)
        totals = out.sum(dim=1)[:, None]
        totals[totals == 0] = 1
    
        return out / totals
        
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
            sizes = [self._n_inputs] + [hp.n_hidden_lin] * hp.block_depth
            sizes = sizes + [self._n_outputs if final else self._n_inputs]
            blocks.append(get_block(sizes, final=final))

        return blocks
    
    @property
    def _n_inputs(self) -> int:
        """
        A `Bulknet` accepts a latitude and profiles for mean wind, buoyancy
        frequency profile, and bulk momentum for each phase speed bin. Each
        profile has `config.n_grid - 1` values.
        """

        return 1 + (config.n_grid - 1) * (2 + hp.n_bins)

    @property
    def _n_outputs(self) -> int:
        """
        For each wavenumber quadrant, a `BulkNet` predicts one momentum profile
        for each phase speed bin, as well as a prediction of the dissipative
        momentum loss at each level.
        """

        return (config.n_grid - 1) * (1 + hp.n_bins)
