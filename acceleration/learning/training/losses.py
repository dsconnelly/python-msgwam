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

        self._scales = torch.sqrt((Y ** 2).mean(dim=0))
        self._scales = _topdown_cummax(self._scales)

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

def _topdown_cummax(a: torch.Tensor) -> torch.Tensor:
    """
    Get the maximum value seen starting from the top of a vertical profile.

    Parameters
    ----------
    a
        Flattened tensor of vertical profiles.

    Returns
    -------
    torch.Tensor
        Tensor of the same length as `a` with cumulative maxima.

    """

    a = torch.flip(a.reshape(-1, config.n_grid - 1), dims=(-1,))
    a = torch.flip(torch.cummax(a, dim=-1)[0], dims=(-1,))

    return a.flatten()
