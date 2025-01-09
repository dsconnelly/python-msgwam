import torch, torch.nn as nn

class FluxLoss(nn.Module):
    """
    Loss module for training `Surrogates` that only enforce physical constraints
    at inference time.
    """

    def forward(
        self,
        targets: torch.Tensor,
        output: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate the mean squared error. Avoids penalizing entries where the
        evaluation-time flux will be clamped to the correct value.

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

        errors = targets - output

        if self.training:
            under = (targets <= 0) & (output <= 0)
            over = (1 <= targets) & (1 <= output)
            errors = errors[~(under | over)]

        return (errors ** 2).mean()
