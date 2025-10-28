from typing import Optional

import torch, torch.nn as nn

from optuna.trial import Trial

from ...hyperparameters import training as hp

class FluxLoss(nn.Module):
    _scales_Y: torch.Tensor
    _scales_W: torch.Tensor
    
    def __init__(self, trial: Trial, W: torch.Tensor) -> None:
        """
        Initialize the loss module.

        Parameters
        ----------
        trial
            Current trial. Used to sample a bias applied to the `Y` loss during
            training epochs.
        W
            Tensor of amplitude parameters in the training dataset, used to
            calculate an array of scales to weight the `W` loss by.

        """

        super().__init__()
        self._bias_Y = trial.suggest_float('bias_Y', 0.05, 0.95)
        self.register_buffer('_scales_W', torch.std(torch.log(W), axis=0))

    def forward(
        self,
        Y: torch.Tensor,
        W: torch.Tensor,
        Y_hat: torch.Tensor,
        W_hat: torch.Tensor,
        reduce: bool=True
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Calculate the loss.

        Parameters
        ----------
        Y, Y_hat
            True and network-predicted shape profiles.
        W, W_hat
            True and network-predict amplitudes.
        reduce
            Whether to take the mean over all entries (so that the gradient can
            be calculated) or to preserve the array structure (for plotting).

        Returns
        -------
        torch.Tensor, torch.Tensor
            Losses for the `Y` and `W` predictions, possibly reduced.
        torch.Tensor
            If `reduce`, the combined loss for this epoch; otherwise `None`.
        
        """

        W = torch.log(W)
        scales_Y, _ = abs(Y).max(dim=-1, keepdim=True)
        scales_Y = hp.loss_scale_Y * scales_Y

        loss_Y = ((Y - Y_hat) / scales_Y) ** 2
        loss_W = ((W - W_hat) / self._scales_W) ** 2

        if reduce:
            loss_Y, loss_W = loss_Y.mean(), loss_W.mean()
            return loss_Y, loss_W, self._combine(loss_Y, loss_W)
        
        return loss_Y, loss_W, None

    def _combine(
        self,
        loss_Y: torch.Tensor,
        loss_W: torch.Tensor
    ) -> torch.Tensor:
        """
        Combine the shape and amplitude losses to obtain one number that can be
        used for backpropagation or evaluation. During training epochs, the loss
        may be biased towards one part or the other, but at evaluation time the
        losses are weighted equally.

        Parameters
        ----------
        loss_Y, loss_W
            Reduced losses for shape and amplitude.

        Returns
        -------
        torch.Tensor
            Combined loss, possibly biased.

        """

        bias = self._bias_Y if self.training else 0.5
        return bias * loss_Y + (1 - bias) * loss_W
