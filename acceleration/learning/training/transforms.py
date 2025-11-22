from typing import Literal

import numba as nb
import numpy as np
import torch, torch.nn as nn

_Array = np.ndarray | torch.Tensor

class Transform(nn.Module):
    _log_scale: torch.Tensor
    _logit_p: torch.Tensor
    _shift: torch.Tensor

    def __init__(
        self,
        a: torch.Tensor,
        learnable: bool,
        has_shift: bool,
        by_bin_only: bool,
        ps: int | tuple[int, int],
    ) -> None:
        """
        Initialize a transform, possibly with learnable parameters. Flexible
        enough to work for all input and output types.

        Parameters
        ----------
        a
            Training data. Used to calculate statistics, which are either the
            statistics used at transform time if `not learnable`, or the initial
            values of those statistics otherwise.
        learnable
            Whether the statistics should be learnable or fixed.
        has_shift
            Whether to include a shift or just a scale.
        by_bin_only
            Whether the shift and scale parameters should be per-level and per-
            phase speed bin or per-bin only.
        ps
            If a tuple the bounds on the order of the root to use at transform
            time. If an integer, the upper and lower bounds are the same. Must
            be an integer if `not learnable`.
        
        """

        super().__init__()

        if isinstance(ps, int):
            ps = (ps, ps)
        elif not learnable:
            raise ValueError('Must specify a power for a fixed Transform')
        
        p_min, p_max = ps
        self._p_min = p_min
        self._dp = p_max - p_min
        a = take_root(a, (p_min + p_max / 2))

        if by_bin_only:
            b = a.permute([0, a.ndim - 1, *range(1, a.ndim - 1)])
            scale = nonzero_stat(b.flatten(0, 1).numpy(), mode='std')
            scale = torch.as_tensor(scale)[..., None]
        else:
            scale = a.std(dim=0)
            scale[scale == 0] = 1

        if has_shift:
            shift = a.mean(dim=0)
        else:
            shift = torch.zeros(1)

        shape = (*a.shape[1:-1], 1)
        logit_p = torch.zeros(shape)

        word = 'parameter' if learnable else 'buffer'
        register = getattr(self, f'register_{word}')
        log_scale = torch.log(scale)

        if learnable:
            log_scale = nn.Parameter(log_scale)
            logit_p = nn.Parameter(logit_p)

        self.register_buffer('_shift', shift)
        register('_log_scale', log_scale)
        register('_logit_p', logit_p)

    @property
    def _p(self) -> torch.Tensor:
        """Get the order of the roots."""

        return self._p_min + self._dp * torch.sigmoid(self._logit_p)
    
    @property
    def _scale(self) -> torch.Tensor:
        """Get the scale parameter to use after taking a root."""

        return torch.exp(self._log_scale)

    def forward(self, a: torch.Tensor, inverse: bool=False) -> torch.Tensor:
        """
        Transform a tensor, or revers the transformation.
        """

        if inverse:
            scale = self._scale
            scale = scale * (scale < 1)
            out = scale * a + self._shift

            return torch.sign(out) * torch.abs(out) ** self._p
        
        return (take_root(a, self._p) - self._shift) / self._scale

@nb.njit
def apply_smoothing(a: np.ndarray) -> np.ndarray:
    """
    Apply a Shapiro filter along the last dimension, while respecting initial
    zeros and so not polluting levels below the source.

    Parameters
    ----------
    a
        Array to filter.

    Returns
    -------
    a
        Array filtered along the last dimension.

    """

    out = np.zeros_like(a)
    for idx in np.ndindex(a.shape[:-1]):
        start = np.argmax(a[idx] != 0)

        for k in range(start, a.shape[-1]):
            out[*idx, k] += a[*idx, max(k - 1, start)]
            out[*idx, k] += a[*idx, min(k + 1, a.shape[-1] - 1)]
            out[*idx, k] += 2 * a[*idx, k]

    return out / 4

def nonzero_stat(a: np.ndarray, mode=Literal['mean', 'std']) -> np.ndarray:
    """
    Take the mean or standard deviation along the outermost axis, including only
    nonzero values.

    Parameters
    ----------
    a
        Array to calculate on.
    mode
        Whether to take the mean or standard deviation.

    Returns
    -------
        Nonzero means or standard deviations, of shape `a.shape[1:]`. Entries
        corresponding to columns that entirely zero are themselves zero.
    
    """

    a[a == 0] = np.nan
    out = np.zeros(a.shape[1:])
    valid = (~np.isnan(a)).sum(axis=0) > 0

    func = getattr(np, f'nan{mode}')
    out[valid] = func(a[:, valid], axis=0)
    a[np.isnan(a)] = 0

    return out

def reshape_data(
    a: _Array,
    n_bins: int,
    mode: Literal['sum', 'skip']
) -> _Array:
    """
    Reshape arrays to have the appropriate number of phase speed bins. Supports
    both `numpy` arrays and `torch` tensors.

    Parameters
    ----------
    a
        Array to reshape, with phase speed bins as the second to last dimension.
    n_bins
        Desired number of phase speed bins in the output.
    mode
        Whether to combine bins by summing (for momentum or vertical fluxes) or
        by taking boundary values (for horizonal fluxes).
    
    Returns
    -------
    _Array
        Array of the appropriate shape.

    """

    if a.shape[-2] == n_bins:
        return a

    if mode == 'sum':
        shape = (*a.shape[:-2], n_bins, -1, a.shape[-1])
        return a.reshape(*shape).sum(-2)
    
    elif mode == 'skip':
        n_skip = a.shape[-2] // n_bins
        return a[..., (n_skip - 1)::n_skip, :]

def take_root(a: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """
    Take a root while respecting the sign of the input.

    Parameters
    ----------
    a
        Data to transform.
    p
        Order of the root to take.

    Returns
    -------
    torch.Tensor
        Root of each element in `a`.

    """

    return torch.sign(a) * abs(a) ** (1 / p)