from typing import Optional

import torch, torch.nn as nn

def standardize(
    a: torch.Tensor,
    means: Optional[torch.Tensor]=None,
    stds: Optional[torch.Tensor]=None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    
    """

    if means is None:
        means = a.mean(dim=0)

    if stds is None:
        stds = a.std(dim=0)

    sdx = stds > 0
    output = torch.zeros_like(a)
    output[:, sdx] = (a - means)[:, sdx] / stds[sdx]

    return output, means, stds

def xavier_init(a: nn.Module | torch.Tensor) -> None:
    """
    Apply Xavier initialization a linear layer or a weight matrix.

    Parameters
    ----------
    a
        Module or weight matrix to potentially initialize. If an `nn.Linear` is
        passed, its weight matrix will be extracted and initialized.

    """

    if isinstance(a, nn.Linear):
        a = a.weight

    if isinstance(a, torch.Tensor):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(a, gain=gain)
