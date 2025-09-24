import torch, torch.nn as nn

from torch.utils.data import DataLoader

class BulkLoss(nn.Module):
    def __init__(self, loader_tr: DataLoader) -> None:
        """
        At initialization, a `BulkLoss` stores the target standard deviations
        in the training data to use when computing losses.

        Parameters
        ----------
        DataLoader
            Loader containing a `TensorDataset` of training data.

        """

        super().__init__()
        *_, M, cg = loader_tr.dataset.tensors
        self._stds = [M.std(dim=0), cg.std(dim=0)]
        self._keeps = [std > 0 for std in self._stds]

    def forward(
        self,
        M: torch.Tensor,
        cg: torch.Tensor,
        M_hat: torch.Tensor,
        cg_hat: torch.Tensor
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

        Ys, Y_hats, loss = [M, cg], [M_hat, cg_hat], 0
        for Y, Y_hat, std, keep in zip(Ys, Y_hats, self._stds, self._keeps):
            error = (Y - Y_hat)[:, keep] / std[keep]
            loss = loss + (error ** 2).mean()

        return loss / 2
