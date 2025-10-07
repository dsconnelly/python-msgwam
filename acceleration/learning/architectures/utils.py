from typing import Literal

import torch, torch.nn as nn

def apply_blocks(
    blocks: nn.ModuleList,
    X: torch.Tensor,
    skip_mode: int
) -> torch.Tensor:
    """
    Apply the blocks of this network, with skip connections either additive,
    concatenative, or nonexistent.

    Parameters
    ----------
    X
        Stacked input features.
    skip_mode
        How to apply skip connections. -1 and 1 correspond to concatenative and
        additive, respectively, while 0 indicates no skip connections (in which
        case the set of blocks is equivalent to one block).
    
    Returns
    -------
    torch.Tensor
        Output of final neural network block.

    """

    output = X
    for block in blocks[:-1]:
        output = block(output)

        if skip_mode == -1:
            output = torch.hstack((output, X))

        elif skip_mode == 1:
            output = output + X

    return blocks[-1](output)

def get_block(
    sizes: list[int],
    batch_norm_pos: int,
    activation: Literal['relu', 'leaky', 'tanh'],
    final: bool=False
) -> nn.Sequential:
    """
    Build a block that will constitute a component of a `BulkNet`.

        Parameters
        ----------
        sizes
            Sizes of each layer.
        batch_norm_pos
            Where to put batch normalization layers. -1 and 1 indicate before
            and after the ReLU, respectively, while 0 indicates omission.
        activation
            What activation to use. If `final`, then the last layer needs to be
            non-negative, and so the last activation will be replaced with a
            ReLU regardless of this choice.
        final
            Whether this is the last block in the network, in which case the
            last output needs to be non-negative definite.

        Returns
        -------
        nn.Sequential
            Module containing the layers in the block.

    """

    cls = {
        'relu' : nn.ReLU,
        'leaky' : nn.LeakyReLU,
        'tanh' : nn.Tanh
    }[activation]

    args = []
    for (a, b) in zip(sizes[:-1], sizes[1:]):
        args = args + [nn.Linear(a, b), cls()]

        if batch_norm_pos != 0:
            k = len(args) - (batch_norm_pos == -1)
            args.insert(k, nn.BatchNorm1d(b))

    if final:
        while not isinstance(args[-1], cls):
            args = args[:-1]

        args = args[:-1] + [nn.ReLU()]

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
