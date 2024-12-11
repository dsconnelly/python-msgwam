from abc import ABC, abstractmethod

import torch, torch.nn as nn

from msgwam import config
from msgwam.dispersion import get_omega_hat

from .. import hyperparameters as hp

from .utils import xavier_init

class SourceNet(nn.Module, ABC):
    """
    `SourceNet` is an abstract class for neural networks that make predictions
    using data available at the gravity wave source: the zonal wind profile and
    the spectral properties of the ray volumes to be launched. This base class
    provides functions for defining common architectures, along with shared
    preprocessing and standardization operations.
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
        Apply the whole forward model, including preprocessing, standardization,
        application of the neural network layers, and postprocessing. Subclasses
        must proide their own `_postprocess` implementations.

        Parameters
        ----------
        u
            Two-dimensional array of zonal wind profiles, whose first dimension
            ranges over samples and whose second dimension ranges over vertical
            grid cell centers.
        X
            Two-dimensional array of coarse ray volume properties, whose first
            dimension ranges over samples and whose second dimension ranges over
            individual ray properties k, l, m, dk, dl, dm, and dens.

        Returns
        -------
        torch.Tensor
            Postprocessed neural network output.

        """

        stacked = self._preprocess(u, X)
        stacked = self._standardize(stacked)
        output = self._predict(stacked)

        return self._postprocess(u, X, output)

    def get_extra_state(self) -> dict[str]:
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
        Save the mean and standard deviation information to be used later. Must
        be called before this model can be applied.

        Parameters
        ----------
        u
            Two-dimensional tensor of training zonal wind profiles.
        X
            Two-dimensional tensor of training ray volume properties.

        """

        stacked = self._preprocess(u, X)
        self.means = stacked.mean(dim=0)
        self.stds = stacked.std(dim=0)

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

    @staticmethod
    def _get_block(sizes: list[int], final: bool=False) -> nn.Sequential:
        """
        `SourceNet` instances consist of one or more blocks consisting of fully
        connected layers. Residual connections occur between each block. This
        function constructs a block given a list of the sizes of each layer.

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
        Create the layers of the neural network and pack them in a `ModuleList`
        for later application. The architecture is a feedforward network with
        zero or more residual connections.
        """

        base = hp.n_layers // hp.n_blocks
        extras = hp.n_layers % hp.n_blocks
        lengths = [base] * hp.n_blocks

        for i in range(extras):
            lengths[i] = lengths[i] + 1

        self._blocks = nn.ModuleList()
        for i, length in enumerate(lengths):
            final = i == hp.n_blocks - 1
            n_last = self._n_outputs if final else self._n_inputs
            sizes = [self._n_inputs] + [hp.layer_size] * (length - 1) + [n_last]
            self._blocks.append(self._get_block(sizes, final))

    @property
    def _n_inputs(self) -> int:
        """
        Return the number of input features the network has. All `SourceNet`
        subclasses take in one feature for each grid point in the zonal wind
        profile and one for each ray volume property considered.
        """

        return (config.n_grid - 1) + 3
    
    @property
    @abstractmethod
    def _n_outputs(self) -> int:
        """
        Return the number of output channels the network should have. Left for
        each subclass to implement, as the prediction targets may differ.
        """
        ...

    @abstractmethod
    def _postprocess(
        self,
        u: torch.Tensor,
        X: torch.Tensor,
        output: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply any necessary postprocessing to the network layer output.
        
        Parameters
        ----------
        u
            Zonal wind profiles, as passed to `forward`.
        X
            Ray volume properties, as passed to `forward`.
        output
            Result of applying the neural network layers to `u` and `X`.

        Returns
        -------
        torch.Tensor
            Postprocessed output data.

        """
        ...
        
    def _predict(self, stacked: torch.Tensor) -> torch.Tensor:
        """
        Apply the neural network layers to the preprocessed and standardized
        input data. Provided as a standalone function so that the network itself
        can be called in isolation.

        Parameters
        ----------
        stacked
            Preprocessed and standardized input data, as returned by calling
            `self._preprocess` followed by `self._standardize`.

        Returns
        -------
        torch.Tensor
            Direct output of the neural network layers underlying this model.

        """

        output = stacked
        for block in self._blocks[:-1]:
            output = block(output) + stacked

        return self._blocks[-1](output)

    @staticmethod
    def _preprocess(u: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """
        Extract the appropriate spectral properties, take the logarithm of those
        that require it, and stack all the inputs together in one array. Makes
        the assumption that all ray volumes have positive phase velocity and
        changes the sign of the zonal wind profile accordingly.

        Parameters
        ----------
        u
            Zonal wind profiles, as passed to `forward`.
        X
            Ray volume properties, as passed to `forward`.

        Returns
        -------
        torch.Tensor
            Stacked and preprocessed input data. The columns correspond to the
            zonal wind profile and the ray volume phase velocities, intrinsic
            periods, and logarithm of action density.

        """

        k, l, m, dk, dl, dm, dens = X.T
        log_A = torch.log(dens * dk * dl * dm)
        u = u * torch.sign(k)[:, None]

        omega_hat = get_omega_hat(k, l, m, config.N_ref)
        T_hat = 2 * torch.pi / omega_hat
        cp_x = omega_hat / abs(k)

        return torch.vstack((u.T, cp_x, T_hat, log_A)).T
    
    def _standardize(self, stacked: torch.Tensor) -> torch.Tensor:
        """
        Standardize a tensor of preprocessed input data using the precalculated
        mean and standard deviation tensors.

        Parameters
        ----------
        stacked
            Tensor to standardize.

        Returns
        -------
        torch.Tensor
            Standardized input data.

        """

        sdx = self.stds > 0
        output = torch.zeros_like(stacked)
        output[:, sdx] = (stacked - self.means)[:, sdx] / self.stds[sdx]

        return output