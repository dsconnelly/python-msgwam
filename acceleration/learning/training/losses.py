import torch, torch.nn as nn

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
        self._stds = Y.std(dim=0)
        self._stds[self._stds == 0] = 1

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

        return (((Y - Y_hat) / self._stds) ** 2).mean()
