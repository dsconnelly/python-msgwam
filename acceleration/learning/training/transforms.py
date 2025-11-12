from typing import Literal

import numba as nb
import numpy as np
import torch, torch.nn as nn

_Array = np.ndarray | torch.Tensor

class Transform(nn.Module):
    _shift: torch.Tensor
    _scale: torch.Tensor

    def __init__(
        self,
        a: torch.Tensor,
        mode: str,
        p: torch.Tensor | int=1,
        scale_only: bool=False
    ) -> None:
        """
        Initialize a module that transforms neural network input data.

        Parameters
        ----------
        a
            Tensor from which to derive transform statistics.
        mode
            What kind of transform to perform. Must be `'constant'` or `'z'`.
        p
            Root to take before transforming.
        scale_only
            Whether to only include the scale term. Useful for sign-definite
            target data.

        """

        self._p = p
        a = take_root(a, p)

        if mode == 'constant':
            b = a.permute([0, a.ndim - 1, *range(1, a.ndim - 1)])
            sigma = nonzero_stat(b.flatten(0, 1).numpy(), mode='std')
            sigma = torch.as_tensor(sigma)[..., None]

            shift = sigma * torch.ones(a.shape[1:])
            scale = sigma * torch.ones(a.shape[1:])

        elif mode == 'z':
            shift = a.mean(dim=0)
            scale = a.std(dim=0)

        else:
            raise ValueError(f'Unknown transform mode: {mode}')
        
        if scale_only:
            shift = 0 * shift

        super().__init__()
        self.register_buffer('_shift', shift)
        self.register_buffer('_scale', scale)

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        """
        Transform the input along the first dimension.

        Parameters
        ----------
        a
            Data to transform.
        
        Returns
        -------
        torch.Tensor
            Transformed data.

        """

        a = take_root(a, self._p)
        out = torch.zeros_like(a)

        sdx = self._scale > 0
        out[:, sdx] = (a - self._shift)[:, sdx] / self._scale[sdx]
        
        return out
    
    def inverse(self, a: torch.Tensor) -> torch.Tensor:
        """
        Invert the transform back into dimensional space.

        Parameters
        ----------
        a
            Transformed data.

        Returns
        -------
        torch.Tensor
            Data with transformation inverted.

        """

        return (self._scale * a + self._shift) ** self._p

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

def take_root(a: torch.Tensor, p: int) -> torch.Tensor:
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