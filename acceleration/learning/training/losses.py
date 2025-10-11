import torch, torch.nn as nn

class BulkLoss(nn.Module):
    def forward(
        self,
        Y: torch.Tensor,
        W: torch.Tensor,
        Y_hat: torch.Tensor,
        W_hat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate the mean-squared loss in momentum and sink profiles. The loss
        is decomposed into shape and scale parameters, so that the entries in
        each sub-profile are taken to sum to unity (or be zero everywhere.)

        Parameters
        ----------
        Y, Y_hat
            True and network-predicted bulk momentum and sink profiles.
        W, W_hat
            True and network-predicted amplitudes for each profile.

        """

        error_shape = ((Y - Y_hat) ** 2).mean()
        error_scale = ((W - W_hat) ** 2).mean()

        return (Y.shape[1] * error_shape + error_scale) / (Y.shape[1] + 1)
