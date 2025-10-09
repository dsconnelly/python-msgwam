from __future__ import annotations
from typing import TYPE_CHECKING

import torch, torch.nn as nn

from msgwam import config

from ...hyperparameters import architectures as hp

from .utils import allocate_layers, apply_blocks, get_block, xavier_init

if TYPE_CHECKING:
    from optuna.trial import Trial

class BulkNet(nn.Module):
    def __init__(self, trial: Trial) -> None:
        """
        At initialization, a `BulkNet` queries the `Trial` object for a number
        of hyperparameters used to instantiate one or more blocks of fully-
        connected network layers.

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
        C: torch.Tensor,
        M: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply the forward model. The inputs should already be transformed before
        being passed to this function.

        Parameters
        ----------
        C
            Tensor of column information, with first dimension ranging over
            samples and second dimension ranging over mean wind, buoyancy
            frequency, and latitude.
        M
            Tensor of bulk momentum profiles, with first dimension ranging over
            samples, second dimension over phase speed bins, and third dimension
            over vertical grid points.

        Returns
        -------
        torch.Tensor, torch.Tensor
            Predicted `Y` and `D` tensors. `D` is the total sink across all
            phase speed bins, while `Y` has the same shape as `M` and contains
            either the updated bulk momentum profiles (if `not learn_delta`) or
            the deltas to those updated profiles.

        """

        X = torch.hstack((C, M.flatten(1, 2)))
        out = apply_blocks(self._blocks, X, self._skip_mode)
        out = out.reshape(-1, self._n_bins + 1, config.n_grid - 1)
        Y, D = out[:, :-1], nn.functional.relu(out[:, -1])
        
        if hp.learn_delta:
            raise NotImplementedError('learn_delta not yet supported')

        else:
            Y = nn.functional.relu(Y)
            total = Y.sum(dim=(1, 2)) + D.sum(dim=1)
            total[total == 0] = 1

            Y, D = Y / total[:, None, None], D / total[:, None]

        return Y, D

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

            args = (self._batch_norm_pos, self._activation, self._dropout_rate)
            blocks.append(get_block(sizes, *args, final=final))

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

        n_hidden = trial.suggest_int('n_hidden', 4, 10)
        n_blocks = trial.suggest_int('n_blocks', 1, min(4, n_hidden))
        self._n_hiddens = allocate_layers(n_hidden, n_blocks)
        self._width = trial.suggest_int('width', 128, 512)

        options = [1, 2, 5]
        i = trial.suggest_int('n_bin_idx', 0, len(options) - 1)
        self._n_bins = options[i]

        if n_blocks > 1:
            args_sm = ('skip_mode', [-1, 1])
            self._skip_mode = trial.suggest_categorical(*args_sm)
        else:
            self._skip_mode = 0

        args_bn = ('batch_norm_pos', [-1, 0, 1])
        args_act = ('activation', ['relu', 'leaky', 'tanh'])

        self._activation = trial.suggest_categorical(*args_act)
        self._batch_norm_pos = trial.suggest_categorical(*args_bn)
        self._dropout_rate = trial.suggest_float('dropout_rate', 0, 0.15)
