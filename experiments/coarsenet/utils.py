from typing import Optional

import numpy as np
import torch
import xarray as xr

def pad_jagged(
    a: np.ndarray | xr.DataArray,
    keep: np.ndarray,
    width: Optional[int]=None
) -> torch.Tensor:
    """
    Given an array `a`, return a tensor containing just those elements indicated
    by the Boolean array `keep`, with each row padded with zeros as necessary.

    Parameters
    ----------
    a
        Array to select from. If an `xr.DataArray`, will be cast to a numpy
        array and assumed to have two dimensions.
    keep
        Boolean array indicating which elements of `a` to keep.
    width
        Width to pad to. Note that if `n` is smaller than the maximum number of
        elements to be kept in a row of `a`, that number will be used instead.

    Returns
    -------
    torch.Tensor
        Padded array of values selected from `a`.

    """

    if isinstance(a, xr.DataArray):
        a = a.values

    per_row = keep.sum(axis=1)
    max_pos = per_row.max()

    if width is not None:
        max_pos = max(max_pos, width)

    out = np.zeros((a.shape[0], max_pos))
    idx = np.arange(max_pos) < per_row[:, None]
    out[idx] = a[keep]

    return torch.as_tensor(out)
