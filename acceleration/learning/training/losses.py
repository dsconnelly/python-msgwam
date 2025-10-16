import numba as nb
import numpy as np
import torch, torch.nn as nn

from .transforms import nonzero_stat

class BulkLoss(nn.Module):
    _scales_W: torch.Tensor

    def __init__(self, _, W: torch.Tensor):
        """
        At initalization, the `BulkLoss` calculates scale parameters from the
        training data with which the losses will be weighted.
        """

        super().__init__()

        b = nonzero_stat(W.cpu().numpy(), 'mean')
        self.register_buffer('_scales_W', torch.as_tensor(b))

    def forward(
        self,
        Y: torch.Tensor,
        W: torch.Tensor,
        Y_hat: torch.Tensor,
        W_hat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate the mean-squared loss in momentum and sink profiles. The loss
        is decomposed into shape and scale parameters, so that the entries in
        each sub-profile are taken to sum to unity (or be zero everywhere.)

        Parameters
        ----------
        Y, Y_hat
            True and network-predicted bulk momentum and sink profiles.
        W, W_hat
            True and network-predicted amplitudes for each profile.

        """

        scales_Y = _get_scales(Y)
        loss_Y = (((Y - Y_hat) / scales_Y) ** 2).mean()
        loss_W = (((W - W_hat) / self._scales_W) ** 2).mean()

        return loss_Y, loss_W

def _get_scales(Y: torch.Tensor, min_frac: float=0.01) -> torch.Tensor:
    """
    Get the scales that should be used to weight each shape profile.

    Parameters
    ----------
    Y
        Tensor of target shape profiles.
    min_frac
        Fraction of the maximum value in each profile a value must exceed to
        be included in the calculation.

    Returns
    -------
    torch.Tensor
        Tensor giving a scale for each profile.

    """

    a = Y.clone()
    ubound, _ = a.max(dim=-1, keepdim=True)
    a[a < min_frac * ubound] = torch.nan

    out = torch.nanmean(a, dim=-1, keepdim=True)
    out = torch.nan_to_num(out)

    out[out == 0] = 1 / Y.shape[1]
    return out