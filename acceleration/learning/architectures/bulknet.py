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
        windN: torch.Tensor,
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

        windN, lat = windN[:, :-1], windN[:, -1:]
        X = torch.hstack((windN, M))
        
        if hp.n_convs > 0:
            shape = (-1, self._n_channels, config.n_grid - 1)
            X = self._conv(X.reshape(*shape)).flatten(1, 2) + X
            X = torch.hstack((X, lat))

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

        if hp.n_convs > 0:
            sizes = [hp.n_hidden_conv] * (hp.n_convs - 1)
            sizes = [self._n_channels] + sizes + [self._n_channels]
            kernels = [max(hp.max_kernel - 2 * i, 3) for i in range(hp.n_convs)]
            self._conv = get_block(sizes, kernels, final=False)

        blocks = nn.ModuleList()
        for i in range(hp.n_blocks):
            final = i == hp.n_blocks - 1
            sizes = [self._n_inputs] + [hp.n_hidden_lin] * hp.block_depth
            sizes = sizes + [self._n_outputs if final else self._n_inputs]
            blocks.append(get_block(sizes, final=final))

        return blocks
    
    @property
    def _n_channels(self) -> int:
        """
        There are channels in the input convolutional layer corresponding to the
        two mean state profiles and a profile for each phase speed bin.
        """

        return 2 + hp.n_bins

    @property
    def _n_inputs(self) -> int:
        """
        A `Bulknet` accepts a latitude and profiles for mean wind, buoyancy
        frequency profile, and bulk momentum for each phase speed bin. Each
        profile has `config.n_grid - 1` values.
        """

        return 1 + (config.n_grid - 1) * self._n_channels

    @property
    def _n_outputs(self) -> int:
        """
        For each wavenumber quadrant, a `BulkNet` predicts one momentum profile
        for each phase speed bin, as well as a prediction of the dissipative
        momentum loss at each level.
        """

        return (config.n_grid - 1) * (1 + hp.n_bins)
