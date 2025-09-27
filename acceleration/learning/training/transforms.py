import torch

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