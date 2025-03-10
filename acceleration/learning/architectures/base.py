from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional

import torch, torch.nn as nn

from msgwam import config

from ...hyperparameters import architectures as hp
from .utils import get_block, get_layer_sizes, xavier_init

class BaseNet(nn.Module, ABC):

    def __init__(self) -> None:
        """
        
        """

        super().__init__()
        self._init_blocks()
        self._init_norms()

        self._online = False
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

        X = self._preprocess(*Xs)

        output = X
        for block in self._blocks[:-1]:
            output = block(output) + X

        return self._blocks[-1](output)
    
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

    @property
    def online(self) -> bool:
        """
        The `online` property determines whether the neural network should
        behave as it would during an online run. Some architectures might not
        have different online behavior.
        """

        return self._online
    
    @online.setter
    def online(self, value: bool) -> None:
        """
        Set the `online` property. If the model is in training mode and `value`
        is `True`, an error will be raised.
        """

        if self.training and value:
            raise ValueError('Cannot switch to online mode in training mode')
        
        self._online = value

    def train(self, mode: bool=True) -> None:
        """
        If the model is put into training mode, the `online` property must be
        set to `False`.
        """

        super().train(mode)

        if mode:
            self._online = False

    def _init_blocks(self) -> None:
        """Initialize the layers of the neural network."""

        ns = get_layer_sizes(flat=True)
        n_in = sum(map(ns.get, self._inputs))
        self._blocks = nn.ModuleList()

        for i in range(hp.n_blocks):
            final = i == hp.n_blocks - 1
            n_out = ns[self._output] if final else n_in
            sizes = [n_in] + [hp.n_hidden] * (hp.n_per_block - 1) + [n_out]
            self._blocks.append(get_block(sizes, final))

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

    def _preprocess(self, *Xs: torch.Tensor) -> torch.Tensor:
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

        to_stack = []
        for name, norm, X in zip(self._inputs, self._norms, Xs):
            X_hat = torch.clone(X)

            if name == 'S':
                X_hat[:, 0] = 2 * torch.pi / X_hat[:, 0]
                X_hat[:, 1] = torch.log(X_hat[:, 1])

            elif name == 'R':
                X_hat[:, 1:3] = 2 * torch.pi / X_hat[:, 1:3]
                X_hat[:, 3] = torch.log(X_hat[:, 3] + 1e-8)

            X_hat = norm(torch.nan_to_num(X_hat))
            to_stack.append(X_hat.flatten(1))

        return torch.hstack(to_stack)
