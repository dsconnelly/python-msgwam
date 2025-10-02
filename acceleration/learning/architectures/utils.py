from typing import Optional

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

def get_block(
    sizes: list[int],
    kernels: Optional[list[int]]=None,
    final: bool=False
) -> nn.Sequential:
    """
    Build a block that will constitute a component of a `BulkNet`.

    Parameters
    ----------
    sizes
        Sizes of each layer. If building a convolutional block, this corresponds
        to the number of channels at each layer.
    kernels
        If `None`, a fully-connected block is built. Otherwise, specifies the
        kernel size at each layer. Should have one fewer element than `sizes`.
    final
        Whether this is the last block in the network, in which case the last
        output needs to be unconstrained output.

    Returns
    -------
    nn.Sequential
        Module containing the layers in the block.

    """

    if kernels is None:
        zipped = zip(sizes[:-1], sizes[1:])
        cls = nn.Linear

    else:
        zipped = zip(sizes[:-1], sizes[1:], kernels)
        cls = lambda *args: nn.Conv1d(*args, padding='same')

    modules = []
    for i, args in enumerate(zipped):
        modules = modules + [cls(*args), nn.ReLU()]
        pre_residual = i == len(sizes) - 2

        if pre_residual or (hp.batch_norm_pos != 0):
            k = len(modules) - (hp.batch_norm_pos == -1)
            modules.insert(k, nn.BatchNorm1d(args[1]))

    accept = type(modules[0]) if final else nn.BatchNorm1d
    i = [i for i, m in enumerate(modules) if isinstance(m, accept)][-1]
    modules = modules[:(i + 1)]

    return nn.Sequential(*modules)

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
