import torch, torch.nn as nn

class BulkLoss(nn.Module):
    def __init__(self, targets: torch.Tensor) -> None:
        """
        At initialization, a `BulkLoss` estimates the scales of the nonzero
        values in the training targets, which will be used to normalize losses.

        Parameters
        ----------
        targets
            Tensor of training targets.

        """

        super().__init__()

        totals = targets.sum(dim=0)
        counts = (targets != 0).sum(dim=0)
        valid = counts > 0

        self._means = torch.ones_like(totals)
        self._means[valid] = totals[valid] / counts[valid]

    def forward(
        self,
        M_hat: torch.Tensor,
        cg_hat: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate the mean-squared loss in momentum and flux profiles.

        Parameters
        ----------
        M, cg
            True bulk momentum and group velocity profiles for each sample.
        M_hat, cg_hat
            Network-output profiles for each sample.

        Returns
        -------
        torch.Tensor
            Mean-squared loss, averaged over both target variables.

        """

        Y_hat = torch.hstack((M_hat, cg_hat))
        return (((targets - Y_hat) / self._means) ** 2).mean()
