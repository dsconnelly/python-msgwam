from __future__ import annotations
from typing import TYPE_CHECKING

import torch, torch.nn as nn

from msgwam import config

from .utils import apply_blocks, get_block, xavier_init

if TYPE_CHECKING:
    from optuna.trial import Trial

class BulkNet(nn.Module):
    def __init__(self, trial: Trial) -> None:
        """
        At initialization, a `BulkNet` creates a series of blocks that will be
        used with skip connections at prediction time.

        Parameters
        ----------
        trial
            Trial from which to draw parameters describing the architecture.

        """

        super().__init__()

        self._set_hyperparameters(trial)
        self._blocks = self._init_blocks()
        self.apply(xavier_init)
        self.to(torch.double)

    def forward(
        self,
        windN: torch.Tensor,
        M: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        torch.Tensor, torch.Tensor
            Tensors of updated bulk momentum profiles and a sink profile.

        """

        X = torch.hstack((windN, M.flatten(1, 2)))
        out = apply_blocks(self._blocks, X, self._skip_mode)

        totals = out.sum(dim=1)[:, None]
        totals[totals == 0] = 1
        out = out / totals

        out = out.reshape(-1, self._n_bins + 1, config.n_grid - 1)
        return out[:, :-1], out[:, -1]

    def _init_blocks(self) -> nn.ModuleList:
        """
        Initialize the main neural network layers.

        Returns
        -------
        nn.ModuleList
            List of blocks to apply at prediction time.

        """

        blocks = nn.ModuleList()
        for i, n_hidden in enumerate(self._n_hiddens):
            final = i == len(self._n_hiddens) - 1
            depth = n_hidden - (not final)

            first = self._n_inputs * (1 + (i > 0 and self._skip_mode == -1))
            last = self._n_outputs if final else self._n_inputs
            sizes = [first] + [self._width] * depth + [last]

            args = (self._batch_norm_pos, self._activation, final)
            blocks.append(get_block(sizes, *args))

        return blocks
    
    @property
    def _n_inputs(self) -> int:
        """
        A `Bulknet` accepts a latitude and profiles for mean wind, buoyancy
        frequency profile, and bulk momentum for each phase speed bin. Each
        profile has `config.n_grid - 1` values.
        """

        return 1 + (config.n_grid - 1) * (2 + self._n_bins)

    @property
    def _n_outputs(self) -> int:
        """
        For each wavenumber quadrant, a `BulkNet` predicts one momentum profile
        for each phase speed bin, as well as a prediction of the dissipative
        momentum loss at each level.
        """

        return (config.n_grid - 1) * (self._n_bins + 1)

    def _set_hyperparameters(self, trial: Trial) -> None:
        """
        Set hyperparameters given a `Trial` object.
        
        Parameters
        ----------
        trial
            Current trial during optimization.

        """

        n_hidden = trial.suggest_int('n_hidden', 4, 16)
        n_blocks = trial.suggest_int('n_blocks', 1, min(5, n_hidden))
        self._n_hiddens = [n_hidden // n_blocks] * n_blocks

        for i in range(n_blocks):
            if sum(self._n_hiddens) == n_hidden:
                break

            self._n_hiddens[i] = self._n_hiddens[i] + 1

        args = ('batch_norm_pos', [-1, 0, 1])
        self._batch_norm_pos = trial.suggest_categorical(*args)
        self._skip_mode = 0

        if n_blocks > 1:
            self._skip_mode = trial.suggest_categorical('skip_mode', [-1, 0, 1])

        activations = ['relu', 'leaky', 'tanh']
        self._activation = trial.suggest_categorical('activation', activations)
        self._n_bins = trial.suggest_categorical('n_bins', [1, 2, 5, 10])
        self._width = trial.suggest_int('width', 128, 512)

        # TODO: uncomment later
        # i = trial.suggest_int('n_bin_idx', 0, 3)
        # self._n_bins = [1, 2, 5, 10][i]