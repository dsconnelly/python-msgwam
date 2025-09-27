import torch, torch.nn as nn

from ...hyperparameters import architectures as hp

def apply_blocks(blocks: nn.ModuleList, X: torch.Tensor) -> torch.Tensor:
    """
    Apply a set of blocks with skip connections after all but the last.

    Parameters
    ----------
    blocks
        List of modules to apply between skip connections.
    X
        Tensor to pass through the blocks.

    Returns
    -------
    torch.Tensor
        Output of final block.

    """

    output = X
    for block in blocks[:-1]:
        output = X + block(output)

    return blocks[-1](output)

def get_block(sizes: list[int], final: bool) -> nn.Sequential:
    """
    Build a block of fully-connected layers for the neural network, placing
    batch normalization layers according to hyperparameter settings.

    Parameters
    ----------
    sizes
        Sizes for each layer of the block.
    final
        Whether this is the last block in the network. If so, the number of
        outputs will be set accordingly and the last layer will be a `ReLU`.
    
    Returns
    -------
    nn.Sequential
        Module containing the resulting layers.

    """

    args = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        args = args + [nn.Linear(a, b), nn.ReLU()]

        if hp.batch_norm_pos != 0:
            k = len(args) - (hp.batch_norm_pos == -1)
            args.insert(k, nn.BatchNorm1d(b))

    if final:
        while not isinstance(args[-1], nn.ReLU):
            args = args[:-1]

    return nn.Sequential(*args)

def xavier_init(layer: nn.Module) -> None:
    """
    Apply Xavier initialization a linear layer.

    Parameters
    ----------
    a
        Module to potentially initialize. If an `nn.Linear` is passed, its
        weight matrix will be initialized.

    """

    if isinstance(layer, nn.Linear):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(layer.weight, gain=gain)
