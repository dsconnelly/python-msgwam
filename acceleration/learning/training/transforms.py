import numba as nb
import numpy as np
import torch

@nb.njit
def apply_smoothing(a: np.ndarray) -> np.ndarray:
    """
    JITted function to apply a Shapiro filter along the last dimension, while
    respecting initial zeros and so not polluting levels below the source.

    Parameters
    ----------
    a
        Array to smooth.

    Returns
    -------
    np.ndarray
        Smoothed array. Zeros in `a` before the first nonzero value in each
        profile will remain zero.

    """

    out = np.zeros_like(a)
    for idx in np.ndindex(a.shape[:-1]):
        start = np.argmax(a[idx] != 0)

        for k in range(start, a.shape[-1]):
            out[*idx, k] += a[*idx, max(k - 1, start)]
            out[*idx, k] += a[*idx, min(k + 1, a.shape[-1] - 1)]
            out[*idx, k] += 2 * a[*idx, k]

    return out / 4

def get_shift_and_scale(
    a: torch.Tensor, mode: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Get shift and scale arrays that can be used to transform an array later.

    Parameters
    ----------
    a
        Tensor for which to calculate shift and scale arrays.
    mode
        What kind of transform to prepare. Can be `'minmax'`, which will cause
        each column to lie in [-1, 1]; `'robust'`, which shifts by the median
        and scales by the IQR; or `'z'`, which shifts by the mean and scales by
        the standard deviation. Can also pass `'none'`, in which case the shift
        and scale will be zero and one, respectively.
    
    Returns
    -------
    torch.Tensor, torch.Tensor
        Shift and scale arrays, respectively.

    """

    if mode == 'minmax':
        mins, _ = a.min(dim=0)
        maxs, _ = a.max(dim=0)

        return (mins + maxs) / 2, (maxs - mins) / 2
    
    if mode == 'none':
        shift = torch.zeros(a.shape[1], dtype=a.dtype)
        scale = torch.ones(a.shape[1], dtype=a.dtype)

        return shift, scale

    if mode == 'robust':
        q25 = torch.quantile(a, 0.25, dim=0)
        q75 = torch.quantile(a, 0.75, dim=0)
        
        return torch.quantile(a, 0.5, dim=0), q75 - q25
    
    if mode == 'z':
        return a.mean(dim=0), a.std(dim=0)
    
    if mode == 'nonzero':
        b = a.clone().numpy()
        b[b == 0] = np.nan

        shift = np.zeros(b.shape[1])
        scale = np.zeros(b.shape[1])

        valid = (~np.isnan(b)).sum(0) > 0
        shift[valid] = np.nanmean(b[:, valid], axis=0)
        scale[valid] = np.nanstd(b[:, valid], axis=0)

        shift = torch.nan_to_num(torch.as_tensor(shift))
        scale = torch.nan_to_num(torch.as_tensor(scale))

        return shift, scale
    
    raise ValueError(f'Unknown transform mode: {mode}')

def transform(
    a: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor
) -> torch.Tensor:
    """
    Transform an array with precomputed shift and scale terms.

    Parameters
    ----------
    a
        Tensor to transform.
    shift, scale
        Tensors as returned by `get_shift_and_scale`.

    Returns
    -------
    torch.Tensor
        Transformed tensor.
    
    """

    valid = scale > 0
    out = torch.zeros_like(a)
    out[:, valid] = (a - shift)[:, valid] / scale[valid]

    return out