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
        output = block(output)

        if hp.skip_mode == -1:
            output = torch.hstack((output, X))

        elif hp.skip_mode == 1:
            output = output + X

    return blocks[-1](output)

def get_block(sizes: list[int], final: bool=False) -> nn.Sequential:
    """
    Build a block that will constitute a component of a `BulkNet`.

    Parameters
    ----------
    sizes
        Sizes of each layer. If building a convolutional block, this corresponds
        to the number of channels at each layer.
    final
        Whether this is the last block in the network, in which case the last
        output needs to be unconstrained output.

    Returns
    -------
    nn.Sequential
        Module containing the layers in the block.

    """

    args = []
    for (a, b) in zip(sizes[:-1], sizes[1:]):
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
