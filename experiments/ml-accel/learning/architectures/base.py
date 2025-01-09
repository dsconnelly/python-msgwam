from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any

import torch, torch.nn as nn

from msgwam import config

from ... import hyperparameters as hp

from .utils import standardize, xavier_init

class SourceNet(nn.Module, ABC):
    """
    Abstract base class for architectures accepting zonal wind and source ray
    volume properties as inputs.
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

    def forward(self, u: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """
        Apply the whole forward model, including standardization, application of
        the neural network layers, and postprocessing.

        Parameters
        ----------
        u
            Three-dimensional array of zonal wind profiles whose first dimension
            ranges over samples, whose second dimension is of length two and
            ranges over historical snapshots, and whose third dimension ranges
            over vertical grid points.
        X
            Two-dimensional array of source ray volume properties.

        Returns
        -------
        torch.Tensor
            Postprocessed neural network output.

        """

        u, X = self._standardize(u, X)
        output = self._predict(u, X)

        return self._postprocess(output)

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

    def get_extra_state(self) -> dict[str, Any]:
        """
        Return the state related to input standardization.

        Returns
        -------
        dict[str]
            Dictionary containing mean and standard deviation tensors.

        """

        return {'means' : self.means, 'stds' : self.stds}
    
    def init_stats(self, u: torch.Tensor, X: torch.Tensor) -> None:
        """
        Save the means and standard deviations to be used later. Must be called
        before `_standardize` can be called.

        Parameters
        ----------
        u
            Tensor of zonal wind profiles, including past snapshots. The zonal
            wind statistics will be calculated over both the current wind and
            the historical profiles.
        X
            Tensor of source ray volume properties.

        """

        args = [u.flatten(0, 1), X]
        self.means = [a.mean(dim=0) for a in args]
        self.stds = [a.std(dim=0) for a in args]

    def set_extra_state(self, state: dict[str]) -> None:
        """
        Set the state related to input standardization.

        Parameters
        ----------
        state
            Dictionary containing mean and standard deviation tensors.

        """

        self.means = state['means']
        self.stds = state['stds']

    def _init_layers(self) -> None:
        """
        Create the layers of the neural network, as specified by the loaded set
        of hyperparameters. This includes the convolutional part to process the
        wind, the dense layers to process the ray volume properties, and the
        dense layers to process the combined hidden states.
        """

        dropout_rate = hp.architectures.dropout_rate

        channels = [2, 8, 2]
        kernel_sizes = [11, 5]
        args = []

        for a, b, size in zip(channels[:-1], channels[1:], kernel_sizes):
            conv = nn.Conv1d(a, b, kernel_size=size, padding='same')
            args.extend([conv, nn.ReLU(), nn.Dropout(dropout_rate)])

        args.append(_SeqBatchNorm(config.n_grid - 1))
        self._conv = nn.Sequential(*args)

        sizes, args = [3, 32, 3], []
        for a, b in zip(sizes[:-1], sizes[1:]):
            args.extend([nn.Linear(a, b), nn.ReLU(), nn.Dropout(dropout_rate)])

        args.append(nn.BatchNorm1d(3))
        self._dense = nn.Sequential(*args)

        n_first = 2 * (config.n_grid - 1) + 3
        sizes = [n_first, 256, self._n_outputs]
        args = []

        for a, b in zip(sizes[:-1], sizes[1:]):
            args.extend([nn.Linear(a, b), nn.ReLU(), nn.Dropout(dropout_rate)])

        self._shared = nn.Sequential(*args[:-2])

    @property
    @abstractmethod
    def _n_outputs(self) -> int:
        """
        Return the number of outputs the final neural network layer should have.
        """
        ...

    @abstractmethod
    def _postprocess(self, output: torch.Tensor) -> torch.Tensor:
        """
        Apply any necessary postprocessing to the neural network output.

        Parameters
        ----------
        output
            Tensor of outputs from the neural network layers.

        Returns
        -------
        torch.Tensor
            Postprocessed output data.

        """
        ...

    def _predict(self, u: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """
        Apply the neural network to the standardized input data.

        Parameters
        ----------
        u
            Standardized zonal wind data.
        X
            Standardized source ray volume properties.

        Returns
        -------
        torch.Tensor
            Output of the neural network layers underlying this model.

        """

        p = self._conv(u) + u
        q = self._dense(X) + X

        output = torch.hstack((p.flatten(1, 2), q))
        return self._shared(output)

    def _standardize(
        self,
        u: torch.Tensor,
        X: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Standardize input wind profiles and ray volume properties to zero mean
        and unit variance.

        Parameters
        ----------
        u
            Tensor of zonal wind profiles.
        X
            Tensor of source ray volume properties.

        """

        u = u.flatten(0, 1)
        u, *_ = standardize(u, self.means[0], self.stds[0])
        X, *_ = standardize(X, self.means[1], self.stds[1])

        return u.reshape(X.shape[0], 2, -1), X
    
class _SeqBatchNorm(nn.BatchNorm1d):
    """
    Batch norm variant that normalizes over the sequence dimension rather than
    over the channel dimension when the input is three-dimensional.
    """

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        """
        Call the parent class implentation with the last two dimensions swapped,
        then swap the dimensions back before returning.
        """

        return super().forward(a.transpose(1, 2)).transpose(1, 2)