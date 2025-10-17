from typing import Callable, Iterator, Literal

import numba as nb
import numpy as np
import torch

_Array = np.ndarray | torch.Tensor
Transform = Callable[[_Array], _Array]

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

def make_transform(a: np.ndarray, mode: str) -> Transform:
    """
    Make a function that transforms an array. Simply calculates the shift and
    scale and returns a reusable function that applies them.

    Parameters
    ----------
    a
        Array to transform.
    mode
        What kind of transform to prepare.

    Returns
    -------
    _Transform
        Function that applies the appropriate shift and scale.
    
    """

    if mode == 'constant':
        b = a.transpose(0, 2, 1).reshape(-1, a.shape[1])
        sigma = nonzero_stat(b, mode='std')[:, None]

        shift = sigma * np.ones(a.shape[1:])
        scale = sigma * np.ones(a.shape[1:])

    elif mode == 'nonzero':
        shift = nonzero_stat(a, 'mean')
        scale = nonzero_stat(a, 'std')
    
    elif mode == 'robust':
        q25 = np.quantile(a, 0.25, axis=0)
        q75 = np.quantile(a, 0.75, axis=0)

        shift = np.quantile(a, 0.5, axis=0)
        scale = q75 - q25

    elif mode == 'z':
        shift = a.mean(axis=0)
        scale = a.std(axis=0)

    else:
        raise ValueError(f'Unknown transform mode: {mode}')      

    shift = torch.as_tensor(shift)
    scale = torch.as_tensor(scale)
    valid = scale > 0

    def transform(b: _Array) -> _Array:
        """
        Apply the shift and scale. Written with several precautions so as to
        both work on `numpy` arrays and be comptabible with `torch.jit`.
        """

        p, q = shift, scale
        if isinstance(b, np.ndarray):
            p = p.numpy()
            q = q.numpy()

        out = 0 * b
        out[:, valid] = (b - p)[:, valid] / q[valid]

        return out

    return transform

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

def reshape_data(n_bins: int, *arrays: _Array) -> Iterator[_Array]:
    """
    Reshape arrays to have the appropriate number of phase speed bins. Designed
    to support both `numpy` arrays and `torch` tensors.

    Parameters
    ----------
    n_bins
        How many bins the output arrays should have.
    tensors
        Arrays to reshape, with first dimensions ranging over samples, second
        dimensions ranging over phase speed bins, and third dimensions ranging
        over vertical grid points. `n_bins` must divide the existing number of
        phase speed bins.
    
    """

    a, *_ = arrays
    shape = (a.shape[0], n_bins, -1, a.shape[2])
    func = lambda a: a.reshape(*shape).sum(2)

    return map(func, arrays)