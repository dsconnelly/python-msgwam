import torch, torch.nn as nn

from .. import hyperparameters as hp
from .utils import apply_basis

class FluxLoss(nn.Module):
    """
    Flexible loss module for training `Surrogate` models, both those that learn
    the fluxes directly and those that learn proxies instead.
    """

    def forward(
        self,
        targets: torch.Tensor,
        output: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate the mean squared error. If the model is learning proxies, we
        convert to actual flux profiles so that the scores are comparable to
        those of the unconstrained models.

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

        if hp.architectures.basis_type != 'none':
            output = apply_basis(output)

        return ((targets - output) ** 2).mean()
