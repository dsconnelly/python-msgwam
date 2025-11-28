from typing import Literal

import numba as nb
import numpy as np
import torch, torch.nn as nn

_Array = np.ndarray | torch.Tensor

class Transform(nn.Module):
    _p: torch.Tensor
    _shift: torch.Tensor
    _scale: torch.Tensor

    def __init__(
        self,
        a: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        has_shift: bool=True,
        by_bin_only: bool=True,
        p: int | torch.Tensor = 1
    ) -> None:
        """
        Initialize a transform.

        Parameters
        ----------
        a
            Tensor from which to calculate scale and shift statistics.
        has_shift
            Whether to include a shift, or just a scale.
        by_bin_only
            If `True`, the second-to-last dimension is interpreted as the phase
            speed bin dimension, and one statistic is used for each entire bin,
            instead of calculating statistics for each level.
        p
            Power to use in the transform. If a `Tensor`, must broadcast to the
            shape of `a`.

        """

        super().__init__()
        p = torch.as_tensor(p)

        if isinstance(a, tuple):
            shift, scale = a

        else:
            a = take_root(a, p)
            shift = has_shift * a.mean(dim=0)

            if by_bin_only:
                b = a.permute([0, a.ndim - 1, *range(1, a.ndim - 1)])
                scale = nonzero_stat(b.flatten(0, 1).numpy(), mode='std')
                scale = torch.as_tensor(scale)[..., None]

            else:
                scale = a.std(dim=0)
                scale[scale == 0] = 1

        self.register_buffer('_p', p)
        self.register_buffer('_shift', shift)
        self.register_buffer('_scale', scale)

    def forward(self, a: torch.Tensor, inverse: bool=False) -> torch.Tensor:
        """
        Apply (or invert) the transform.

        Parameters
        ----------
        a
            Data to (un)transform.
        inverse
            Whether the transform should be inverted.

        Returns
        -------
        torch.Tensor
            (Un)transformed data.

        """

        if inverse:
            scale = self._scale * (self._scale < 1)
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
    mode: Literal['coarsen', 'from_left']
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
    
    Returns
    -------
    _Array
        Array of the appropriate shape.

    """

    if a.shape[-2] == n_bins:
        return a
    
    if mode == 'coarsen':
        shape = (*a.shape[:-2], n_bins, -1, a.shape[-1])
        return a.reshape(*shape).sum(-2)
    
    elif mode == 'from_left':
        p = a[..., :(n_bins - 1), :]
        q = a[..., (n_bins - 1):, :]
        q = q.sum(-2)[..., None, :]

        func = torch.cat if isinstance(a, torch.Tensor) else np.concatenate
        return func((p, q), -2)
    
    raise ValueError(f'Unknown reshape mode {mode}')

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