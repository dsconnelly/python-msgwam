from typing import Literal, Optional

import torch, torch.nn as nn

class FluxLoss(nn.Module):
    _scales_Y: torch.Tensor
    
    def __init__(
        self,
        loss_type: Literal['mse', 'smae'],
        Y: torch.Tensor
    ) -> None:
        """
        Initialize the loss module.

        Parameters
        ----------
        trial
            Current trial. Used to sample a bias applied to the `Y` loss during
            training epochs.
        Y
            Tensor of training targets.

        """

        super().__init__()

        if loss_type not in ['mse', 'smae']:
            raise ValueError(f'Unknown loss function {loss_type}')

        self._loss_type = loss_type
        scales_Y = torch.clamp(torch.std(Y, dim=(0, 3)), min=0.1)
        self.register_buffer('_scales_Y', scales_Y[:, :, None])

    def forward(
        self,
        Y: torch.Tensor,
        Y_hat: torch.Tensor,
        reduce: bool=True
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Calculate the loss.

        Parameters
        ----------
        Y, Y_hat
            True and network-predicted shape profiles.
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

        loss_Y = Y - Y_hat

        if self._loss_type == 'mse':
            loss_Y = loss_Y ** 2

        elif self._loss_type == 'smae':
            loss_Y = _smae(loss_Y)

        if reduce:
            loss_Y = loss_Y.mean()

        return loss_Y

def _smae(error: torch.Tensor) -> torch.Tensor:
    """
    Calculate the smoothed mean absolute error. Behaves like `abs(error)` when
    `error` is large, and `error ** 2` when it is small.

    Parameters
    ----------
    error
        (Potentially scaled) differences between prediction and target.

    Returns
    -------
    torch.Tensor 
        Loss values.

    """

    return error * torch.tanh(error)
