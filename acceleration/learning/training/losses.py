from typing import Literal, Optional

import numpy as np
import torch, torch.nn as nn

from optuna.trial import Trial
from torch.linalg import vector_norm

from ...hyperparameters import training as hp

from .reconstruction import get_dM
from .transforms import Transform

class FluxLoss(nn.Module):
    _scales_Y: torch.Tensor
    _weights: torch.Tensor
    
    def __init__(
        self,
        trial: Trial,
        Y_trans: Transform
    ) -> None:
        """
        Initialize the loss module.

        Parameters
        ----------
        trial
            Current trial.
        Y_trans
            Transform to apply and invert on network outputs.

        """

        super().__init__()

        self._Y_trans = Y_trans
        use_weights = trial.suggest_categorical('use_weights', [True, False])
        self._use_weights = use_weights

        if use_weights:
            weights = self._sample_weights(trial)
            self.register_buffer('_weights', weights)

    def forward(
        self,
        Y: torch.Tensor,
        Y_hat: torch.Tensor,
        reduce: bool=True
    ) -> torch.Tensor:
        """
        Calculate the loss.

        Parameters
        ----------
        Y
            True (untransformed) targets concatenated with `dM` profiles.
        Y_hat
            Network outputs in the transformed space.
        reduce
            Whether to take the mean over all entries (so that the gradient can
            be calculated) or to preserve the array structure (for plotting).

        Returns
        -------
        torch.Tensor
            Loss on the `dM` profiles, possibly concatenated with loss on data
            in the transformed space.
        
        """

        dM_hat = get_dM(self._Y_trans(Y_hat, inverse=True))
        loss = _vnorm_loss(Y[:, -1], dM_hat, eps=1e-4)

        if self.training and self._use_weights:
            Y_t = self._Y_trans(Y[:, :-1], inverse=False)
            loss_t = _vnorm_loss(Y_t, Y_hat, eps=0.01)

            loss = torch.cat((loss[:, None], loss_t), dim=1)
            loss = (self._weights * loss).sum(1)

        if not reduce:
            return loss
        
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

        return torch.as_tensor([a, b - a, 1 - b])[:, None]

def _vnorm_loss(
    a: torch.Tensor,
    a_hat: torch.Tensor,
    eps: float
) -> torch.Tensor:
    """
    Calculate the norm of the error vector divided by the norm of the target,
    with a small number floor to avoid division by zero.
    """

    norms = vector_norm(a, dim=-1)
    norms = torch.clamp(norms, min=eps)

    return vector_norm(a - a_hat, dim=-1) / norms

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
