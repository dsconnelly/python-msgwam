from __future__ import annotations
from abc import ABC, abstractmethod

import torch, torch.nn as nn

from msgwam import config

from .. import hyperparameters as hp

from .standardizer import StandardizerMixin
from .utils import xavier_init

class SourceNet(nn.Module, StandardizerMixin, ABC):
    """
    Abstract base class for architectures with residual connections. Requires
    subclasses to implement a postprocessing function.
    """

    def __init__(self) -> None:
        """
        Create the neural network layers, apply Xavier initialization to their
        weight matrices, and set the model to use double precision.
        """

        super().__init__()
        self._init_layers()
        self.apply(xavier_init)
        self.to(torch.double)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Apply the whole forward model, including standardization, application of
        the neural network layers, and postprocessing.

        Parameters
        ----------
        X
            Two-dimensional array of input features.

        Returns
        -------
        torch.Tensor
            Postprocessed neural network output

        """

        output = self._predict(self._standardize(X, 0))
        return self._postprocess(X, output)

    @classmethod
    def from_name(cls, name: str) -> SourceNet:
        """
        Instantiate a `SourceNet` subclass from its name.

        Parameters
        ----------
        name
            Name of the subclass to instantiate.

        Returns
        -------
        SourceNet
            Instantiated neural network.

        """

        subs = cls.__subclasses__()
        i = [s.__name__ for s in subs].index(name)
        return subs[i]()

    @staticmethod
    def _get_block(sizes: list[int], final: bool=False) -> nn.Sequential:
        """
        Construct a block of fully connected layers. Residual connections will
        presumably be applied between blocks of the type returned here.

        Parameters
        ----------
        sizes
            Size of each hidden layer in the block.
        final
            Whether or not this block will be the last in the network, in which
            case the block will end with a bare linear layer.

        Returns
        -------
        nn.Sequential
            Constructed block of fully connected layers.

        """

        args = []
        for a, b in zip(sizes[:-1], sizes[1:]):
            args.append(nn.Linear(a, b))

            if hp.batch_norm_pos == -1:
                args.append(nn.BatchNorm1d(b))

            args.append(nn.ReLU())

            if hp.batch_norm_pos == 1:
                args.append(nn.BatchNorm1d(b))

            args.append(nn.Dropout(hp.dropout_rate))

        if final:
            n_drop = 2 + abs(hp.batch_norm_pos)
            args = args[:-n_drop]

        return nn.Sequential(*args)
    
    def _init_layers(self) -> None:
        """
        Create the layers of the neural network, as specified by the loaded set
        of hyperparameters, and pack them in a `ModuleList`. This implementation
        creates the blocks between residual connections, though subclasses with
        more complex behavior may extend this function.
        """

        self._blocks = nn.ModuleList()
        length = hp.layers_per_block - 1
        
        for i in range(hp.n_blocks):
            final = i == hp.n_blocks - 1
            n_last = self._n_final if final else self._n_inputs
            sizes = [self._n_inputs] + [hp.layer_size] * length + [n_last]
            self._blocks.append(self._get_block(sizes, final))

    @property
    @abstractmethod
    def _n_final(self) -> int:
        """
        Return the number of outputs the final block should have. In most cases,
        this corresponds to the number of outputs the network has.
        """
        ...

    @property
    def _n_inputs(self) -> int:
        """
        Return the number of input features each block has. All subclasses have
        blocks taking in one feature for each value in the zonal wind profile
        and one for each ray volume property considered.
        """

        return (config.n_grid - 1) + 3

    def _predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Apply the neural network to the standardized input data. Includes the
        basic application of the blocks and residual connections. Subclasses
        with more complex behavior can override or extend this function.

        Parameters
        ----------
        X
            Standardized input data.

        Returns
        -------
        torch.Tensor
            Output of the neural network layers underlying this model.

        """

        output = X
        for block in self._blocks[:-1]:
            output = block(output) + X

        return self._blocks[-1](output)

    @abstractmethod
    def _postprocess(
        self,
        X: torch.Tensor,
        output: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply any postprocessing to the network layer output.

        Parameters
        ----------
        X
            Tensor of input features.
        output
            Tensor of outputs obtained by applying the neural network to `X`.

        Returns
        -------
        torch.Tensor
            Postprocessed output data.

        """
        ...