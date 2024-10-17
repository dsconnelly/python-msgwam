from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch, torch.nn as nn

from msgwam import config
from msgwam.constants import PROP_NAMES
from msgwam.dispersion import cg_r, cp_x, m_from
from msgwam.utils import put

import hyperparameters as params
from utils import xavier_init

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

class SourceNet(nn.Module, ABC):
    """
    `SourceNet` is an abstract class for neural networks that make predictions
    using information available at the gravity wave source; namely, the zonal
    wind profile and the properties of ray volumes to be launched. This class
    provides functions for defining common architectures as well as shared
    preprocessing and standardization operations.
    """

    props_in = ['k', 'm', 'dm', 'dens']
    idx_in = [PROP_NAMES.index(prop) for prop in props_in]

    def __init__(self) -> None:
        """
        Create the neural network layers, apply Xavier initialization to their
        weights, and set the model to use double precision.
        """

        super().__init__()
        self._init_layers()
        self.apply(xavier_init)
        self.to(torch.double)

    def forward(self, u: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Apply the neural network, including input and output processing. The
        data pipeline consists of preprocessing, standardization, application of
        the layers themselves, and postprocessing. Each subclass must provide
        its own `_postprocess` implementation.

        Parameters
        ----------
        u
            Two-dimensional tensor of zonal wind profiles whose second dimension
            ranges over vertical grid points.
        Y
            Two-dimensional tensor whose second dimension ranges over fine or
            coarse ray volume properties.

        Returns
        -------
        torch.Tensor
            Postprocessed output.

        """

        stacked = self._preprocess(u, Y)
        stacked = self._standardize(stacked)

        output = self._predict(stacked)
        output = self._postprocess(u, Y, output)

        return output

    def get_extra_state(self) -> dict[str]:
        """
        Return the state related to input standardization.

        Returns
        -------
        dict[str]
            Dictionary containing mean and standard deviation tensors.

        """

        return {'means' : self.means, 'stds' : self.stds}

    def init_stats(self, u: torch.Tensor, Y: torch.Tensor) -> None:
        """
        Save the mean and standard deviation information to be used later. Must
        be called before this model can be used.

        Parameters
        ----------
        u
            Two-dimensional tensor of training zonal wind profiles.
        Y
            Two-dimensional tensor of training ray volume properties.

        """

        stacked = self._preprocess(u, Y)
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

    def _get_block(self, sizes: list[int], final: bool=False) -> nn.Sequential:
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
            case a special last layer might be applied.

        """

        args = []
        for a, b in zip(sizes[:-1], sizes[1:]):
            args.append(nn.Linear(a, b))

            if params.batch_norm_pos == -1:
                args.append(nn.BatchNorm1d(b))

            args.append(nn.ReLU())

            if params.batch_norm_pos == 1:
                args.append(nn.BatchNorm1d(b))

            args.append(nn.Dropout(params.dropout_rate))

        if final:
            n_drop = 2 + abs(params.batch_norm_pos)
            args = args[:-n_drop] + [self._get_last_layer()]

        return nn.Sequential(*args)

    def _init_layers(self) -> None:
        """
        Create the layers of the neural network and pack them in a `ModuleList`
        for later application. The architecture is a standard feedforward
        network, potentially with one or more residual connections.
        """

        base = params.network_size // params.n_blocks
        extras = params.network_size % params.n_blocks
        lengths = [base] * params.n_blocks

        for i in range(extras):
            lengths[i] = lengths[i] + 1

        self.blocks = nn.ModuleList()
        for i, length in enumerate(lengths):
            final = i == params.n_blocks - 1
            n_last = self._n_outputs if final else self._n_inputs

            sizes = [self._n_inputs] + [256] * length + [n_last]
            self.blocks.append(self._get_block(sizes, final))

    @abstractmethod
    def _get_last_layer(self) -> nn.Module:
        """
        Get the last layer of the neural network. Different subclasses might use
        different last layers to handle different types of output data.

        Returns
        -------
        nn.Module
            Last layer to apply to the neural network. Likely an activation.

        """
        ...

    @property
    def _n_inputs(self) -> int:
        """
        Return the number of input features the network has. All `SourceNet`
        subclasses take in one feature for each grid point in the zonal wind
        profile and one for each ray volume property considered.
        """
        
        return config.n_grid - 1 + len(self.props_in)

    @property
    @abstractmethod
    def _n_outputs(self) -> int:
        """Return the number of output channels the network should have."""
        ...

    @abstractmethod
    def _postprocess(
        self,
        u: torch.Tensor,
        Y: torch.Tensor,
        output: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply any necessary postprocessing to the output of the network layers.
        Each subclass must provide an implementation to be called in `forward`.

        Parameters
        ----------
        u
            Zonal wind profiles, as passed to `forward`.
        Y
            Ray volume properties, as passed to `forward`.
        output
            Result of applying the neural network layers to `u` and `Y`.

        Returns
        -------
        torch.Tensor
            Postprocessed output data.

        """
        ...

    def _predict(self, stacked: torch.Tensor) -> torch.Tensor:
        """
        Apply the neural network layers themselves to the preprocessed input
        data. Provided as a separate function so that the neural network itself
        can be called in isolation.

        Parameters
        ----------
        stacked
            Standardized, preprocessed, and combined input data, as returned by
            `self._preprocess` followed by `self._standardize`.

        Returns
        -------
        torch.Tensor
            Result of applying the neural network, before postprocessing.

        """

        output = stacked
        for block in self.blocks[:-1]:
            output = block(output) + stacked

        return self.blocks[-1](output)

    @classmethod
    def _preprocess(cls, u: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Replace coarse ray volume properties with the logarithms of their
        absolute values, and flip the wind profiles as necessary.

        Parameters
        ----------
        u
            Zonal wind profiles, as passed to `forward`.
        Y
            Ray volume properties, as passed to `forward`.

        Returns
        -------
        torch.Tensor
            Stacked and preprocessed input data.

        """

        signs = torch.sign(Y[:, 2])[:, None]
        Y = torch.log(abs(Y[:, cls.idx_in]))

        return torch.hstack((signs * u, Y))
    
    @property
    def _sdx(self) -> torch.Tensor:
        """
        Return indices where the training data has nonzero standard deviation.
        """

        return self.stds > 0
    
    def _standardize(self, stacked: torch.Tensor) -> torch.Tensor:
        """
        Standardize a tensor of preprocessed wind and ray volume data.

        Parameters
        ----------
        stacked
            Tensor to standarize.

        Returns
        -------
        torch.Tensor
            Standardized input data.

        """

        output = torch.zeros_like(stacked)
        means, stds, sdx = self.means, self.stds, self._sdx
        output[:, sdx] = (stacked - means)[:, sdx] / stds[sdx]

        return output

class Surrogate(SourceNet):
    """
    `Surrogate` accepts a zonal wind profile along with a set of ray volume
    properties and predicts the time-mean nondimensional momentum flux profile
    associated with that ray volume over the integration period.
    """

    def _get_last_layer(self) -> nn.Module:
        """
        `Surrogate` does not have an activation on the last layer. The network
        is allowed to produce negative values during training, but the outputs
        are constrained to [0, 1] at inference time.
        """

        return nn.Identity()

    @property
    def _n_outputs(self) -> int:
        """
        `Surrogate` has an output for each vertical grid point. The fluxes are
        predicted on the cell faces, so we can just use `config.n_grid`.
        """

        return config.n_grid
    
    def _postprocess(
        self,
        _: torch.Tensor,
        Y: torch.Tensor,
        output: torch.Tensor
    ) -> torch.Tensor:
        """
        `Surrogate` models choose the appropriate sign for each flux profile
        based on the zonal wavenumber of each sample. When making inferences,
        the predictions are clamped to fall between zero and one, so that when
        multiplied by the scaling factor they both are sign-definite and
        respect momentum conservation.
        """

        if not self.training:
            output = torch.clamp(output, min=0, max=1)

        signs = torch.sign(Y[:, 2])[:, None]
        return signs * output

class Tinkerer(SourceNet):
    """
    `Tinkerer` accepts a zonal wind profile and coarse ray volume properties,
    and returns the properties of a coarse ray volume that should better mimic
    the online behavior of the constituent fine ray volumes.
    """

    @staticmethod
    def _get_flux(Y: torch.Tensor) -> torch.Tensor:
        """
        Return the momentum flux associated with each ray in a batch.

        Parameters
        ----------
        Y
            Two-dimensional tensor of ray volume properties.

        Returns
        -------
        torch.Tensor
            One-dimensional tensor giving the (signed) momentum flux associated
            with each ray in `Y`.

        """

        *_, k, l, m, dk, dl, dm, dens = Y.T
        flux = k * (dens * dk * dl * dm) * cg_r(k, l, m)

        return flux

    def _get_last_layer(self) -> nn.Module:
        """
        Because `Tinkerer` makes predictions in the normalized wave property
        space, its last layer outputs can be positive or negative.
        """

        return nn.Identity()
    
    @property
    def _n_outputs(self) -> int:
        """
        `Tinkerer` has an output for each property it modifies.
        """

        return 1

        return len(self.props_in)
    
    def _postprocess(
        self,
        _: torch.Tensor,
        Y: torch.Tensor,
        output: torch.Tensor
    ) -> torch.Tensor:
        """
        Take the neural network outputs from the nondimensional and standardized
        space back to dimensional ray volume property space. Makes sure to fix
        the signs of the wavenumbers as necessary. If in evaluation mode, maybe
        fix the output ray volume to have the same momentum flux as the input,
        according to the value of `params.conservative`.
        """

        return output
    
    def _online_postprocess(
        self,
        u: torch.Tensor,
        Y: torch.Tensor,
        output: torch.Tensor
    ) -> torch.Tensor:
        """
        
        """

        cp = cp_x(*Y.T[2:5])
        m = m_from(*Y.T[2:4], cp + output.flatten())
        output = put(Y.T, 4, m).T

        # signs = torch.sign(Y[:, self.idx_in])
        # Y_proc = self._preprocess(u, Y)[:, -4:]
        # output = signs * torch.exp(Y_proc + output)

        # means, stds = self._Y_stats
        # signs = torch.sign(Y[:, self.idx_in])
        # output = signs * torch.exp(output * stds + means)
        # output = put(Y.T, self.idx_in, output.T).T

        if params.conservative:
            factors = self._get_flux(Y) / self._get_flux(output)
            output = put(output.T, -1, factors * output[:, -1]).T

        return output

    @property
    def _Y_stats(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return the saved means and standard deviations corresponding only to
        the ray volume properties.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Tensors of means and standard deviations for the (nondimensional)
            properties in `self.props_in`.

        """

        return self.means[-self._n_outputs:], self.stds[-self._n_outputs:]
