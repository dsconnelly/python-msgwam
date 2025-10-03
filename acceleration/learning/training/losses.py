import numba
import torch, torch.nn as nn

from msgwam import config

class BulkLoss(nn.Module):
    def __init__(self, Y: torch.Tensor) -> None:
        """
        At initialization, a `BulkLoss` estimates the scales of the nonzero
        values in the training targets, which will be used to normalize losses.

        Parameters
        ----------
        Y
            Tensor of training targets.

        """

        super().__init__()

        scales = torch.sqrt((Y ** 2).mean(dim=0))
        scales = scales.reshape(-1, config.n_grid - 1)
        self._scales = _topdown_fill(scales).flatten()

    def forward(self, Y: torch.Tensor, Y_hat: torch.Tensor) -> torch.Tensor:
        """
        Calculate the mean-squared loss in momentum and flux profiles.

        Parameters
        ----------
        Y, Y_hat
            True and network-predicted momentum and dissipation terms.

        Returns
        -------
        torch.Tensor
            Mean-squared loss, scaled by precomputed nonzero means.

        """

        return (((Y - Y_hat) / self._scales) ** 2).mean()

def _topdown_fill(a: torch.Tensor) -> torch.Tensor:
    """
    Fill in zeros in a profile with the lowest nonzero value.

    Parameters
    ----------
    a
        Tensor whose first dimension ranges over profiles and whose second
        dimension ranges over the vertical grid.
    
    Returns
    -------
    torch.Tensor
        Tensor with leading zeros filled in.

    """

    mask = (a > 0).int()
    firsts = a[torch.arange(a.shape[0]), mask.argmax(dim=1)]
    return a + firsts[:, None] * (1 - mask)
