import torch.nn as nn

from msgwam import config

from ...hyperparameters import architectures as hp
from ..preprocessing import get_overrides

def get_block(sizes: list[int], final: bool) -> nn.Sequential:
    """
    Build a block of fully connected layers for the neural network, placing
    batch normalization and dropout layers correctly.

    Parameters
    ----------
    sizes
        List of hidden layer sizes.
    final
        Whether this is the last block in the network.

    Returns
    -------
    nn.Sequential
        Module containing the specified layers.

    """

    args = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        args = args + [
            nn.Linear(a, b), nn.ReLU(),
            nn.Dropout(hp.dropout_rate)
        ]
        
        if hp.batch_norm_pos != 0:
            k = (hp.batch_norm_pos - 5) // 2
            args.insert(k, nn.BatchNorm1d(b))

    if final:
        while not isinstance(args[-1], nn.Linear):
            args = args[:-1]

    return nn.Sequential(*args)

def get_layer_sizes(flat: bool) -> dict[str, int]:
    """
    Calculate the size that an input or output layer should have depending on
    what kinds of data it is to represent. Provided as a function because the
    `config` module is not initially available.

    Parameters
    ----------
    flat
        Whether to return values for flattened input arrays (e.g for actual
        fully-connected layers) or on a per-channel basis (for normalization).

    """

    with config.override(**get_overrides()):
        values = {
            'u' : config.n_grid - 1,
            'S' : 2,
            'R' : 5,
            'Z' : hp.n_latent,
            'F' : config.n_grid
        }

        if flat:
            values['S'] = values['S'] * config.n_source
            values['R'] = values['R'] * config.n_max

    return values

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
