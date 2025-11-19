from typing import Iterator, Literal

import torch, torch.nn as nn

ACTIVATIONS = {
    'relu' : nn.ReLU,
    'leaky' : nn.LeakyReLU,
    'tanh' : nn.Tanh
}

def allocate_layers(n_layers: int, n_blocks: int) -> list[int]:
    """
    Allocate a specified number of layers between the requested numbeer of
    blocks, such that the larger blocks come first.

    Parameters
    ----------
    n_layers
        How many total layers there should be.
    n_blocks
        How many blocks those layers should be split among.

    Returns
    -------
    list[int]
        How many layers each block should have.

    """

    base, rem = divmod(n_layers, n_blocks)
    out = [base + (i < rem) for i in range(n_blocks)]

    return out

def apply_blocks(
    blocks: nn.ModuleList,
    X: torch.Tensor,
    skip_mode: int
) -> torch.Tensor:
    """
    Apply the blocks of a network, with either additive, concatenative, or
    nonexistent skip connections.

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
    dropout_rate: float,
    final: bool=False
) -> nn.Sequential:
    """
    Build a block that will constitute a component of a `BulkNet`. Each block
    consists of several fully-connected layers and activation functions, along
    with possible batch normalization and dropout layers. The precise structure
    of the block is set by the keyword arguments.

    Parameters
    ----------
    sizes
        Sizes of each layer. The resulting block will have `len(sizes) - 1`
        fully-connected layers.
    batch_norm_pos
        Where to put batch normalization layers. -1 and 1 indicate before
        and after the activation, respectively, while 0 indicates omission.
    activation
        What activation to use. If `final` and `not hp.learn_delta`, then
        the last layer needs to be non-negative, and so the last activation
        will be replaced with a ReLU regardless of this choice.
    dropout
        Dropout rate to use. If zero, the dropout layers will have no effect,
        but they are included anyway.
    final
        Whether this is the last block in the network. If so, the last layer of
        the block should be a `Linear` layer, so that the post-processing in the
        `BulkNet` forward function can work properly.

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

        args = args + [nn.Dropout(dropout_rate)]

    if final:
        while not isinstance(args[-1], nn.Linear):
            args = args[:-1]

    return nn.Sequential(*args)

def iter_pairs(values: list[int]) -> Iterator[tuple[int, int]]:
    """
    Iterate over pairs of adjacent values in a list.

    Parameters
    ----------
    values
        List of values to iterate.

    Yields
    ------
    tuple[int, int]
        Adjacent pairs of values. Yields one fewer pair than there are values in
        the provided list.

    """

    for a, b in zip(values[:-1], values[1:]):
        yield a, b

def maybe_interp(a: torch.Tensor, n: int) -> torch.Tensor:
    """
    Interpolate along the last dimension of a Tensor, with a check to ensure
    that the dimension is not already the correct length.

    Parameters
    ----------
    a
        Tensor to interpolate.
    n
        Desired length for the last dimension.
    
    Returns
    -------
    torch.Tensor
        Interpolated data.

    """

    if a.shape[-1] == n:
        return a

    return nn.functional.interpolate(a, n, mode='linear', align_corners=False)

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
