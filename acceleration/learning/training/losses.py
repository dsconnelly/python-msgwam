import numpy as np
import torch, torch.nn as nn

from optuna.trial import Trial

from ..propagators import EulerianPropagator
from .reconstruction import get_dM, cg_from_T_hat
from .transforms import Transform, nonzero_stat

class VelocityLoss(nn.Module):
    _cpt: torch.Tensor
    _scales: torch.Tensor
    _weights: torch.Tensor
    
    def __init__(self, trial: Trial, Y: torch.Tensor):
        """
        Initialize various buffers and parameters.
        """

        super().__init__()

        _, edges_cpt = EulerianPropagator._init_edges(1, Y.shape[-2])
        cpt = torch.as_tensor((edges_cpt[:-1] + edges_cpt[1:]) / 2)
        self.register_buffer('_cpt', cpt[:, None])

        Y = Y * (Y[:, 1] > 1e-14)[:, None]
        Y = Y.permute([0, 3, 1, 2]).flatten(0, 1)
        scales = nonzero_stat(Y.cpu().numpy(), mode='std')[..., None]
        self.register_buffer('_scales', torch.as_tensor(scales))

        w_T = trial.suggest_float('w_T', 0, 1)
        weights = torch.as_tensor([w_T, 1 - w_T])
        self.register_buffer('_weights', weights[:, None, None])
        
    def forward(
        self,
        Nf: torch.Tensor,
        Y: torch.Tensor,
        T_nn: torch.Tensor,
        reduce: bool=True
    ) -> torch.Tensor:
        """
        Calculate the loss, either a weighted combination of the period and the
        group velocity or just the latter at evaluation time.
        """

        N, f = Nf[:, None, :-1], Nf[:, -1, None, None]
        cg_nn = cg_from_T_hat(N, f, self._cpt, T_nn)
        Y_hat = torch.stack((T_nn, cg_nn), dim=1)

        mask = (Y[:, 1] > 1e-14)[:, None]
        loss = mask * ((Y - Y_hat) / self._scales) ** 2

        if self.training:
            loss = (self._weights * loss).sum(dim=1)

        else:
            loss = loss[:, 1]

        if reduce:
            return loss.mean()
        
        return loss

class FluxLoss(nn.Module):
    _scales: torch.Tensor
    _weights: torch.Tensor
    
    def __init__(
        self,
        trial: Trial,
        Y: torch.Tensor,
        Y_trans: Transform,
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
        self.register_buffer('_weights', weights)
        self._skew = trial.suggest_float('skew', 0, 4)

        scales_t = Y[:, :-1].std(dim=(0, -1))[..., None]
        scales_dM = abs(Y[:, -1]).mean(dim=(0, -1))[..., None]
        scales = torch.cat((scales_t, scales_dM[None]), dim=0)
        self.register_buffer('_scales', scales)

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
        loss = _smae((Y[:, -1] - dM_hat) / self._scales[-1])[:, None]

        if self.training or for_plotting:
            loss_t = ((Y[:, :-1] - Y_hat) / self._scales[:-1]) ** 2

            if for_plotting:
                return torch.cat((loss_t, loss), dim=1)

            maxes = abs(Y[:, :-1]).amax(dim=-1, keepdim=True)
            maxes = torch.where(maxes > 0, maxes, 1)

            rescale = abs(Y[:, :-1]) / maxes
            rescale = self._skew * rescale + 1
            loss_t = (rescale ** (self._Y_trans._p - 1)) * loss_t

            loss = torch.cat((loss_t, loss), dim=1)
            loss = (self._weights * loss).sum(dim=1)

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
