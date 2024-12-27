from typing import Optional

import numpy as np
import torch

def nanstd(a: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    """
    
    """

    kwargs['axis'] = kwargs.pop('dim', None)
    return torch.as_tensor(np.nanstd(a.numpy(), *args, **kwargs))

def standardize(
    a: torch.Tensor,
    means: Optional[torch.Tensor]=None,
    stds: Optional[torch.Tensor]=None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Standardize a tensor along the first dimension.

    Parameters
    ----------
    a
        Tensor containing data to standardize.
    means
        Means to use during standardization. If `None`, the mean along the first
        dimension will be computed and used.
    stds
        Standard deviations to use during standardization. If `None`, the
        standard deviation along the first dimension will be computed and used.

    Returns
    -------
    torch.Tensor
        Standardized data.
    torch.Tensor, torch.Tensor
        Means and standard deviations used during standardization. If either of
        these statistics was provided, they will be returned as is.

    """

    if means is None:
        means = a.mean(dim=0)

    if stds is None:
        stds = a.std(dim=0)

    shape = a.shape
    if len(shape) > 2:
        a = a.flatten(1, -1)
        means = means.flatten(0, -1)
        stds = stds.flatten(0, -1)

    sdx = stds > 0
    output = torch.zeros_like(a)
    output[:, sdx] = (a - means)[:, sdx] / stds[sdx]

    if len(shape) > 2:
        output = output.reshape(shape)
        means = means.reshape(shape[1:])
        stds = stds.reshape(shape[1:])

    return output, means, stds