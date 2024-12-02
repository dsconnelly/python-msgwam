import torch.nn as nn

def xavier_init(layer: nn.Module) -> None:
    """
    Apply Xavier initialization to a layer if it is an `nn.Linear`.

    Parameters
    ----------
    layer
        Module to potentially initialize.

    """

    if isinstance(layer, nn.Linear):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(layer.weight, gain=gain)
