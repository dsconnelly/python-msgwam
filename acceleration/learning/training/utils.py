from typing import Optional

import torch

from .io import MultifileDataset

def get_flux_statistics(
    dataset: MultifileDataset
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    
    """

    to_stack = []
    for i in range(dataset.n_tasks):
        to_stack.append(dataset._load('F', i))

    Y = torch.vstack(to_stack)
    return Y.mean(dim=0), Y.std(dim=0)

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

    sdx = stds > 0
    output = torch.zeros_like(a)
    output[:, sdx] = (a - means)[:, sdx] / stds[sdx]

    return output, means, stds
