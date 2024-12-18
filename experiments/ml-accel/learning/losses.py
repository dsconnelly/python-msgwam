import torch, torch.nn as nn

from . import hyperparameters as hp
from .utils import apply_basis, standardize

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
            _, self.means, self.stds = standardize(Y)

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
                targets, *_ = standardize(targets, self.means, self.stds)

            else:
                targets = apply_basis(targets)
                output = apply_basis(output)

        return ((targets - output) ** 2).mean()
