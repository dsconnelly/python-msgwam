import numpy as np
import torch, torch.nn as nn

from optuna.trial import Trial

from .reconstruction import get_dM
from .transforms import Transform, nonzero_stat

class FluxLoss(nn.Module):
    _scales_dM: torch.Tensor
    _scales_t: torch.Tensor
    _weights: torch.Tensor
    
    def __init__(
        self,
        trial: Trial,
        Y: torch.Tensor,
        Y_trans: Transform
    ) -> None:
        """
        Initialize the loss module.

        Parameters
        ----------
        trial
            Current trial.
        Y
            Tensor of training targets, to use to calculate statistics.
        Y_trans
            Transform to apply and invert on network outputs.

        """

        super().__init__()

        self._Y_trans = Y_trans
        weights = self._sample_weights(trial)
        dM = Y[:, -1].flatten(0, 1).cpu().numpy()

        scales_t = Y[:, :-1].std(dim=(0, -1))[..., None]
        scales_dM = nonzero_stat(abs(dM), mode='mean')

        scales_dM[-5:] = np.nan
        scales_dM[scales_dM < 1e-8] = np.nan
        scales_dM = np.nan_to_num(1 / scales_dM)
        scales_dM = torch.as_tensor(scales_dM)

        self.register_buffer('_weights', weights)
        self.register_buffer('_scales_t', scales_t)
        self.register_buffer('_scales_dM', scales_dM)

    def forward(
        self,
        Y: torch.Tensor,
        Y_hat: torch.Tensor,
        for_plotting: bool=False
    ) -> torch.Tensor:
        """
        Calculate the loss.

        Parameters
        ----------
        Y
            True (untransformed) targets concatenated with `dM` profiles.
        Y_hat
            Network outputs in the transformed space.
        for_plotting
            Whether to take the mean over all entries (so that the gradient can
            be calculated) or to preserve the array structure (for plotting).

        Returns
        -------
        torch.Tensor
            Loss on the `dM` profiles, possibly concatenated with loss on data
            in the transformed space.
        
        """

        dM_hat = get_dM(self._Y_trans(Y_hat, inverse=True))
        loss = _smae((Y[:, -1] - dM_hat) * self._scales_dM)[:, None]

        if self.training or for_plotting:
            loss_t = ((Y[:, :-1] - Y_hat) / self._scales_t) ** 2

            if for_plotting:
                return torch.cat((loss_t, loss), dim=1)

            loss = torch.cat((loss_t, loss), dim=1)
            loss = (self._weights * loss).sum(dim=1)

        else:
            loss = loss * (loss < 2)

        return loss.mean()
    
    def _sample_weights(self, trial: Trial) -> torch.Tensor:
        """
        Draw a uniform sample from the 3-dimensional probability simplex. Works
        by sampling the maximum and minimum separately.
        """

        u = trial.suggest_float('loss_u', 0, 1)
        v = trial.suggest_float('loss_v', 0, 1)
        b = np.sqrt(v)
        a = u * b

        return torch.as_tensor([a, b - a, 1 - b])[:, None, None]

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
