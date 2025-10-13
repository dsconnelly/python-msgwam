import torch, torch.nn as nn

from .transforms import nonzero_stat

class BulkLoss(nn.Module):
    _scales_Y: torch.Tensor
    _scales_W: torch.Tensor

    def __init__(self, Y: torch.Tensor, W: torch.Tensor):
        """
        At initalization, the `BulkLoss` calculates scale parameters from the
        training data with which the losses will be weighted.
        """

        super().__init__()

        a = nonzero_stat(Y.transpose(1, 2).flatten(0, 1).numpy(), 'std')
        b = nonzero_stat(W[..., 0].numpy(), 'std')

        self.register_buffer('_scales_Y', torch.as_tensor(a)[:, None])
        self.register_buffer('_scales_W', torch.as_tensor(b)[:, None])

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

        error_Y = (((Y - Y_hat) / self._scales_Y) ** 2).mean()
        error_W = (((W - W_hat) / self._scales_W) ** 2).mean()

        return (Y.shape[2] * error_Y + error_W) / (Y.shape[2] + 1)
