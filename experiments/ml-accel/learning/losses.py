import torch, torch.nn as nn

from . import hyperparameters as hp
from .utils import (
    apply_basis,
    get_proxy_statistics,
    standardize,
    transform_proxies
)

class FluxLoss(nn.Module):
    """
    Flexible loss module for training `Surrogate` models, both those that learn
    the fluxes directly and those that learn proxies instead.
    """

    def __init__(self, Y: torch.Tensor) -> None:
        """
        If we are training a constrained `Surrogate`, we need to compute and
        save the training target statistics, so that we can standardize the
        targets whe computing the loss later.

        Parameters
        ----------
        Y
            Two-dimensional tensor of training targets.

        """

        super().__init__()

        if hp.basis_type != 'none':
            self.means, self.stds = get_proxy_statistics(Y)
            self.stds[self.stds == 0] = 1

    def forward(
        self,
        targets: torch.Tensor,
        output: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate the mean squared error. If the model is learning proxies and
        we are in an evaluation step, we convert to actual flux profiles so that
        the scores are comparable to those of the unconstrained models.

        Parameters
        ----------
        targets
            Training or evaluation targets.
        output
            Neural network outputs.

        Returns
        -------
        torch.Tensor
            Mean squared error averaged over all samples and output channels.

        """

        if hp.basis_type != 'none':
            
            if self.training:
                mask = (targets[:, 0] > 0).int()
                errors = (targets - output) / self.stds
                loss = (errors[:, 0] ** 2).sum()

                for i in range(1, 3):
                    loss = loss + ((errors[:, i] * mask) ** 2).sum()

                return loss / (mask.numel() + 2 * mask.sum()).item()

            else:
                error = ((output - targets) / self.stds) ** 2

                keep = output[:, 0] > 0
                error[:, 1][~keep] = torch.nan
                error[:, 2][~keep] = torch.nan

                e1, e2, e3 = torch.nanmean(error, dim=(0, 2))
                print(e1.item(), e2.item(), e3.item())
                
                targets = apply_basis(targets)
                output = apply_basis(output)

        return ((targets - output) ** 2).mean()
