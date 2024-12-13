import torch, torch.nn as nn

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
