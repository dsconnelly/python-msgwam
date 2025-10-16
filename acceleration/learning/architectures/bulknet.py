from __future__ import annotations
from typing import TYPE_CHECKING, Literal

import torch, torch.nn as nn

from msgwam import config

from ...hyperparameters import architectures as hp

from .unet import UNet
from .utils import allocate_layers, apply_blocks, get_block, xavier_init

if TYPE_CHECKING:
    from optuna.trial import Trial

_RELU = nn.functional.relu
_SOFTPLUS = nn.functional.softplus

class BulkNet(nn.Module):
    def __init__(self, trial: Trial, relax_beta: bool) -> None:
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
        self._relax_beta = relax_beta
        self._set_hyperparameters(trial)

        self._updates = 0
        self._beta = 1
        
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

        if self._has_unet:
            C, lat = C[:, :-1], C[:, -1:, None]
            C = C.reshape(-1, 2, config.n_grid - 1)
            lat = lat * torch.ones_like(C[:, :1])

            X = torch.cat((C, lat, M), dim=1)
            X = self._unet(X).flatten(1, 2)

        else:
            X = torch.hstack((C, M.flatten(1, 2)))

        out = apply_blocks(self._blocks, X, self._skip_mode)
        Y, W = out[:, :-self._n_weights], out[:, -self._n_weights:, None]
        Y = Y.reshape(-1, self._n_weights, config.n_grid - 1)

        Y = self._normalize(Y, d=2)
        W = self._normalize(W, d=1)

        return Y, W

    @property
    def progress(self) -> float:
        """
        Float representing the progress towards the final value of beta, used to
        calculate the score during training. If `_beta` is not being relaxed
        (i.e. we are doing hyperparameter search) the progress is always 1.
        """

        if not self._relax_beta:
            return 1

        return (self._beta - 1) / (hp.beta_final - 1)

    def update_beta(self, loss_tr: float) -> None:
        """
        Update the current value of the Softplus sharpness parameter. Should be
        called once per iteration in the training loop.

        Parameters
        ----------
        loss_tr
            Current training loss. When the training loss is sufficiently low,
            the sharpness parameter begins to increase.
        
        """

        if not self._relax_beta:
            return

        if loss_tr > self._ramp_start:
            return
        
        self._updates = self._updates + 1
        factor = min(self._updates / hp.ramp_length, 1) ** hp.ramp_p
        self._beta = 1 + (hp.beta_final - 1) * factor

    def _approx_relu(self, a: torch.Tensor) -> torch.Tensor:
        """
        A function that, during training, applies Softplus with the current
        sharpness parameter `self._beta`, but at evaluation time applies the
        actual ReLU function. As beta is increased, this function becomes more
        like ReLU during training.

        Parameters
        ----------
        a
            Data to transform.

        Returns
        -------
        torch.Tensor
            Tensor of non-negative values.

        """

        if self.training or (not self._relax_beta):
            return _SOFTPLUS(a, self._beta)
        
        return _RELU(a)

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
    
    def _normalize(self, a: torch.Tensor, d: Literal[1, 2]) -> torch.Tensor:
        """
        Make a tensor non-negative and then scale its entries such that its sum
        along a given axis is one, while respecting the physical constraints.

        Parameters
        ----------
        a
            Tensor of fully-connected layer output.
        d
            Axis to normalize along. If `2`, the array is the array of shape
            values, and the function applied to all entries is the approximation
            to the ReLU defined above. If `1`, this is the array of scales, and
            the same function is applied except to the last weight (that for the
            sink profile) which is always positive and thus takes Softplus.
        
        Returns
        -------
        torch.Tensor
            Tensor with no negative entries summing to one.

        """

        if d == 1:
            others = self._approx_relu(a[:, :-1])
            totals = others.sum(dim=1, keepdim=True)
            totals[totals == 0] = 1

            last = hp.max_sink * torch.sigmoid(a[:, -1:])
            others = (1 - last) * others / totals

            return torch.cat((others, last), dim=1)

        out = self._approx_relu(a)
        totals = out.sum(dim=d, keepdim=True)
        totals[totals == 0] = 1

        return out / totals

    @property
    def _n_inputs(self) -> int:
        """
        A `Bulknet` accepts a latitude and profiles for mean wind, buoyancy
        frequency profile, and bulk momentum for each phase speed bin. Each
        profile has `config.n_grid - 1` values.
        """

        if self._has_unet:
            return (config.n_grid - 1) * self._n_weights

        return 1 + (config.n_grid - 1) * (2 + self._n_bins)

    @property
    def _n_outputs(self) -> int:
        """
        For each wavenumber quadrant, a `BulkNet` predicts one momentum profile
        for each phase speed bin, as well as a prediction of the dissipative
        momentum loss at each level. Then there is a weight for each profile.
        """

        return config.n_grid * self._n_weights
    
    @property
    def _n_weights(self) -> int:
        """
        A `Bulknet` outputs one scaling coefficient for each phase speed bin and
        one for the sink profile.
        """

        return self._n_bins + 1

    def _set_hyperparameters(self, trial: Trial) -> None:
        """
        Set hyperparameters given a `Trial` object.
        
        Parameters
        ----------
        trial
            Current trial during optimization.

        """

        options = [1, 2, 5]
        i = trial.suggest_int('n_bin_idx', 0, len(options) - 1)
        self._n_bins = options[i]

        self._has_unet = trial.suggest_categorical('has_unet', [True, False])
        n_hidden = trial.suggest_int('n_hidden', 4, 5 if self._has_unet else 10)

        if self._has_unet:
            self._unet = UNet(self._n_bins, trial)
            n_blocks = 1

        else:    
            n_blocks = trial.suggest_int('n_blocks', 1, min(4, n_hidden))

        self._n_hiddens = allocate_layers(n_hidden, n_blocks)
        self._width = trial.suggest_int('width', 128, 512)

        if n_blocks > 1:
            args_sm = ('skip_mode', [-1, 1])
            self._skip_mode = trial.suggest_categorical(*args_sm)
        else:
            self._skip_mode = 0

        args_bn = ('batch_norm_pos', [-1, 0, 1])
        args_act = ('activation', ['relu', 'leaky', 'tanh'])

        self._activation = trial.suggest_categorical(*args_act)
        self._batch_norm_pos = trial.suggest_categorical(*args_bn)
        self._dropout_rate = trial.suggest_float('dropout_rate', 0, 0.5)

        if self._relax_beta:
            self._ramp_start = trial.suggest_float('ramp_start', 0, 1)