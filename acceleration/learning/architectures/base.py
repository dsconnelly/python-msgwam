from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional

import torch, torch.nn as nn

from msgwam import config

from .utils import get_layer_sizes, xavier_init

class BaseNet(nn.Module, ABC):
    """
    Abstract class for the neural networks used in the ray tracing emulator.
    Provides input preprocessing and normalization as well as useful functions
    at initialization time.
    """

    def __init__(self) -> None:
        """
        Call functions to implement subclass-specific layer construction and
        create input normalization layers, and then set the whole model to use
        double precision and Xavier initialization.
        """

        super().__init__()
        self._init_layers()
        self._init_norms()

        self.apply(xavier_init)
        self.to(torch.double)

    def forward(self, *Xs: torch.Tensor) -> torch.Tensor:
        """
        Apply the whole forward model, including preprocessing, standardization,
        application of the neural network layers, and postprocessing.

        Parameters
        ----------
        Xs
            Input tensors.

        Returns
        -------
        torch.Tensor
            Postprocessed neural network output.

        """

        Xs = self._preprocess(*Xs)
        return self._forward(*Xs)

    @classmethod
    def from_kwargs(cls, *, name: str, tag: Optional[str]=None) -> BaseNet:
        """
        Load a `BaseNet` subclass by name, potentially also loading pretrained
        weights from disk.

        Parameters
        ----------
        name
            Subclass to load.
        tag
            Suffix on file containing trained data, as by `train_networks`.

        Returns
        -------
        BaseNet
            Initialized model, possibly with pretrained weights loaded.
        
        """

        subs = cls.__subclasses__()
        model = [s for s in subs if s.__name__.lower() == name][0]()

        if tag is not None:
            path = f'data/{config.name}/models/{name}-{tag}.pkl'
            model.load_state_dict(torch.load(path, weights_only=True))

        return model
    
    @abstractmethod
    def _forward(self, *Xs: torch.Tensor) -> torch.Tensor:
        """
        Forward function logic, called by `forward` after preprocessing. Each
        subclass must provide an implementation based on the layers created by
        `_init_layers` at initialization.

        Parameters
        ----------
        Xs
            Tensor or tensors of input information. The tensors come with any
            necessary variable transformations applied and normalized.

        Returns
        -------
        torch.Tensor
            Neural network output.

        """
        ...
    
    @abstractmethod
    def _init_layers(self) -> None:
        """
        Initialize the layers of the neural network. Each subclass must provide
        an implementation of this method. Layer initialization is done here and
        not as an override to `__init__` so that it can occur before Xavier
        initialization and proper `dtype` assignment.
        """
        ...

    def _init_norms(self) -> None:
        """Initialize the input normalization layers."""

        ns = get_layer_sizes(flat=False)
        func = lambda s: nn.BatchNorm1d(ns[s], affine=False)
        self._norms = nn.ModuleList(map(func, self._inputs))

    @property
    @abstractmethod
    def _inputs(self) -> list[str]:
        """
        Return the names of the inputs accepted by the network. Each subclass
        must implement this function so that `_preprocess` can transform each
        input tensor correctly.

        Returns
        -------
        list[str]
            List of input types. Can include `'u'`, `'S'`, `'R'`, and `'Z'`.

        """
        ...

    @property
    @abstractmethod
    def _output(self) -> str:
        """
        Return the name of the output returned by the network, which will be
        used to determine the correct number of output nodes.

        Returns
        -------
        str
            Output type. Can be `'R'` or `'Z'`.

        """
        ...

    def _preprocess(self, *Xs: torch.Tensor) -> tuple[torch.Tensor]:
        """
        Preprocess the input tensors for computation, depending on what type of
        data each subclass accepts, and then flatten and stack the data.

        Parameters
        ----------
        Xs
            List of input tensors. Should be the same length as `_inputs`.

        Returns
        -------
        torch.Tensor
            Preprocessed, flattened, and stacked input data.

        """

        if len(Xs) != len(self._inputs):
            msg = f'Expected {len(self._inputs)} input tensors, '
            msg = msg + f'but got {len(Xs)}'
            raise ValueError(msg)

        outputs = []
        for name, norm, X in zip(self._inputs, self._norms, Xs):
            X_hat = torch.clone(X)

            if name == 'S':
                X_hat[:, 0] = 2 * torch.pi / X_hat[:, 0]
                X_hat[:, 1] = torch.log(X_hat[:, 1])

            elif name == 'R':
                idx = torch.argsort(X_hat[:, :1], dim=-1)
                X_hat = torch.take_along_dim(X, idx, dim=-1)
                X_hat[:, 1:3] = 2 * torch.pi / X_hat[:, 1:3]
                X_hat[:, 3] = torch.log(X_hat[:, 3] + 1e-8)

            outputs.append(norm(torch.nan_to_num(X_hat)))

        return tuple(outputs)
