import numba as nb
import numpy as np
import torch, torch.nn as nn

from .transforms import nonzero_std

class BulkLoss(nn.Module):
    _scales_M_tr: torch.Tensor
    _scales_M_ev: torch.Tensor
    _scales_D: torch.Tensor

    def __init__(self, M_out: torch.Tensor, D: torch.Tensor) -> None:
        """
        At initialization, a `BulkLoss` estimates the scales of the nonzero
        values in the training targets, which will be used to normalize losses.

        Parameters
        ----------
        Y
            Tensor of training targets.

        """

        super().__init__()

        names = [f'_scales_{s}' for s in ('M_tr', 'M_ev', 'D')]
        datas = [M_out, M_out.sum(dim=1)[:, None], D[:, None]]
        cpu = torch.device('cpu')

        for name, data in zip(names, datas):
            scales = nonzero_std(data.to(cpu))
            _topdown_fill(scales.numpy())

            self.register_buffer(name, scales)

    def forward(
        self,
        M: torch.Tensor,
        D: torch.Tensor,
        M_hat: torch.Tensor,
        D_hat: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate the mean-squared loss in momentum and flux profiles. During
        training, the loss is computed on each phase speed bin separately, but
        at evaluation time only the total profile is scored.

        Parameters
        ----------
        M, M_hat
            True and network-predicted bulk momentum profiles.
        D, D_hat
            True and network-predicted dissipation profiles.

        Returns
        -------
        torch.Tensor
            Mean-squared loss, scaled by precomputed scales.

        """

        if self.training:
            scales_M = self._scales_M_tr

        else:
            M = M.sum(dim=1)
            M_hat = M_hat.sum(dim=1)
            scales_M = self._scales_M_ev

        loss_M = (((M - M_hat) / scales_M) ** 2).mean()
        loss_D = (((D - D_hat) / self._scales_D) ** 2).mean()
        
        return 0.5 * (loss_M + loss_D)

@nb.njit
def _topdown_fill(a: np.ndarray) -> None:
    """
    Fill an array of profiles that may start with zeros with the lowest nonzero
    value in each profile. Modifies the input in place.

    Parameters
    ----------
    a
        Array to fill in.
    
    """

    for i in range(a.shape[0]):
        j = np.argmax(a[i] != 0)
        a[i, :j] = a[i, j]
