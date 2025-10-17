import numpy as np
import torch, torch.nn as nn

class BulkLoss(nn.Module):
    _threshold: torch.Tensor
    _scales_W: torch.Tensor

    def __init__(self, W: torch.Tensor):
        """
        At initalization, the `BulkLoss` calculates scale parameters from the
        training data with which the losses will be weighted.
        """

        super().__init__()

        threshold, scales_W = _get_W_buffers(W.cpu().numpy())
        self.register_buffer('_threshold', threshold)
        self.register_buffer('_scales_W', scales_W)

    def forward(
        self,
        Y: torch.Tensor,
        W: torch.Tensor,
        Y_hat: torch.Tensor,
        W_hat: torch.Tensor,
        reduce: bool=True
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
        reduce
            Whether to take the average over all components (for training) or to
            leave the bin and vertical dimensions intact (for plotting).

        """

        mask = W > self._threshold
        mask_hat = torch.exp(W_hat) > self._threshold
        W = torch.log(torch.clip(W, min=self._threshold))
        scales_Y = _get_Y_scales(Y)

        loss_Y = mask * (((Y - Y_hat) / scales_Y) ** 2)
        loss_W = ((W - W_hat) / self._scales_W) ** 2
        loss_W = loss_W * (mask | mask_hat).int()

        if reduce:
            return loss_Y.mean(), loss_W.mean()

        return loss_Y, loss_W
    
def _get_W_buffers(W: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Get the buffers needed to calculate `W` errors.

    Parameters
    ----------
    W
        Values of `W` in the training set.

    Returns
    -------
    np.ndarray
        Threshold for each phase speed bin and the sink.
    np.ndarray
        Standard deviation (in log space) for each bin and the sink.

    """

    W[W == 0] = np.nan
    threshold = 0.25 * np.nanmin(W, axis=0)
    threshold = np.maximum(threshold, 0.0001)
    sigma = np.nanstd(np.log(W), axis=0)
    W[np.isnan(W)] = 0

    return torch.as_tensor(threshold), torch.as_tensor(sigma)

def _get_Y_scales(Y: torch.Tensor) -> torch.Tensor:
    """
    Get a scale for each shape profile in the batch.

    Parameters
    ----------
    Y
        Batch of training targets.

    Returns
    -------
    torch.Tensor
        Scale value for each profile in `Y`.

    """

    a = Y.clone()
    ubound, _ = a.max(dim=-1, keepdim=True)
    a[a < 0.01 * ubound] = torch.nan

    out = torch.nanmean(a, dim=-1, keepdim=True)
    out = torch.nan_to_num(out)
    out[out == 0] = 1

    return out