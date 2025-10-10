import numba as nb
import numpy as np
import torch, torch.nn as nn

from ...hyperparameters import architectures as hp

from .transforms import _Array, nonzero_stat

class BulkLoss(nn.Module):
    _scales_Y_tr: torch.Tensor
    _scales_Y_ev: torch.Tensor
    _scales_D: torch.Tensor

    def __init__(self, Y: _Array, D: _Array) -> None:
        """
        At initialization a `BulkLoss` estimates the scales of the nonzero
        values in each training target, which will be used to normalize losses.
        The scale is the mean for deltas and sink profiles, and the standard
        deviation otherwise.

        Parameters
        ----------
        Y, D
            Tensors of training targets.

        """

        super().__init__()
        self._n_bins = Y.shape[1]

        cpu = torch.device('cpu')
        names = ['Y_tr', 'Y_ev', 'D']
        datas = [Y, Y.sum(1), D]

        if isinstance(Y, torch.Tensor):
            datas = [a.to(cpu).numpy() for a in datas]

        for name, data in zip(names, datas):
            mode = 'mean' if (name == 'D' or hp.learn_delta) else 'std'
            scales = nonzero_stat(abs(data), mode=mode)
            _topdown_fill(scales)

            scales = torch.as_tensor(scales)
            self.register_buffer(f'_scales_{name}', scales)

    def forward(
        self,
        Y: torch.Tensor,
        D: torch.Tensor,
        Y_hat: torch.Tensor,
        D_hat: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate the mean-squared loss in momentum and sink profiles. During
        training, the momentum is evaluated on each phase speed bin separately,
        while at evaluation time only the total profile is scored.

        Parameters
        ----------
        Y, Y_hat
            True and network-predicted bulk momentum profiles (or deltas).
        D, D_hat
            True and network-predicted sink profiles.

        """

        if self.training:
            scales_Y = self._scales_Y_tr

        else:
            Y = Y.sum(dim=1)
            Y_hat = Y_hat.sum(dim=1)
            scales_Y = self._scales_Y_ev

        loss_Y = (((Y - Y_hat) / scales_Y) ** 2).mean()
        loss_D = (((D - D_hat) / self._scales_D) ** 2).mean()
        weight = self._n_bins if self.training else 1

        return (weight * loss_Y + loss_D) / (weight + 1)

@nb.njit
def _topdown_fill(a: np.ndarray) -> None:
    """
    Fill a profile or an array of profiles that may start with zeros with the
    lowest nonzero value in each profile. Useful so that levels with no observed
    nonzero values do not produce NaN values during training.

    Parameters
    ----------
    a
        Array to fill in. Will be modified in place.

    """

    for idx in np.ndindex(a.shape[:-1]):
        j = np.argmax(a[*idx] != 0)
        a[*idx, :j] = a[*idx, j]
