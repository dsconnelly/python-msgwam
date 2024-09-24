from abc import ABC, abstractmethod

import torch, torch.nn as nn

from msgwam import config
from msgwam.constants import PROP_NAMES
from msgwam.dispersion import cg_r
from msgwam.utils import put

import hyperparameters as params
from utils import xavier_init

class SourceNet(nn.Module, ABC):
    """
    `SourceNet` is an abstract class for neural networks making predictions
    based on information available at the gravity wave source; namely, the zonal
    wind profile and the properties of (coarse) ray volumes to be launched. The
    preprocessing operations are somewhat specific and thus grouped here.
    """

    props_in = ['k', 'm', 'dm', 'dens']
    idx_in = [PROP_NAMES.index(prop) for prop in props_in]

    def __init__(self) -> None:
        """
        Create the neural network layers, apply Xavier initialization to their
        weights, and set this model to use double precision.
        """

        super().__init__()
        self._init_layers()
        self.apply(xavier_init)
        self.to(torch.double)

    @abstractmethod
    def forward(self, u: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Apply the neural network, including input and output processing. Each
        subclass must define its own implementation, although `_predict` is
        provided to handle the input preprocessing and the application of the
        layers themselves.

        Parameters
        ----------
        u
            Two-dimensional tensor of zonal wind profiles whose second dimension
            ranges over vertical grid points.
        Y
            Two-dimensional tensor whose second dimension ranges over coarse ray
            volume properties.

        Returns
        -------
        torch.Tensor
            Two-dimensional tensor whose second dimension ranges over the
            outputs of the particular subclass.

        """
        ...

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
        be called before this model can be evaluated.

        Parameters
        ----------
        u
            Two-dimensional tensor of training zonal wind profiles.
        Y
            Two-dimensional tensor of training coarse ray volume properties.

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
            n_drop = 3 if params.batch_norm_pos != 0 else 2
            args = args[:-n_drop] + [self._get_last_layer()]

        return nn.Sequential(*args)        

    @abstractmethod
    def _get_last_layer(self) -> nn.Module:
        """
        Get the last layer of the neural network. Must be implemented separately
        by each subclass, because different outputs have different requirements
        and some subclasses might want to choose this layer based on the value
        of `params.last_layer`.

        Returns
        -------
        nn.Module
            Last layer to apply to the neural network, likely an activation.

        """
        ...

    def _init_layers(self) -> None:
        """
        Create the layers of the neural network and pack them in a `Sequential`
        for later application. For now, the architecture is a simple feedforward
        network, with the number of hidden layers set by `params.network_size`.
        """

        block_length = int(params.network_size // params.n_blocks)
        self.blocks = nn.ModuleList()

        for i in range(params.n_blocks):
            final = i == params.n_blocks - 1
            n_last = self._n_outputs if final else self._n_inputs
            sizes = [self._n_inputs] + [256] * block_length + [n_last]
            self.blocks.append(self._get_block(sizes, final=final))

    @property
    def _n_inputs(self) -> int:
        """
        Return the number of input features the network has. `SourceNet` takes
        in one feature for grid point in the zonal wind profile, and one for
        each ray volume property considered.
        """

        return len(self.props_in) + config.n_grid - 1

    @property
    @abstractmethod
    def _n_outputs(self) -> int:
        """
        Return the number of output channels the network should have. Must be
        implemented by each subclass.
        """
        ...

    def _predict(self, stacked: torch.Tensor) -> torch.Tensor:
        """
        Apply the neural network layers. Assumes that the zonal wind and coarse
        ray volume data have already been preprocessed and standardized.

        Parameters
        ----------
        stacked
            Two-dimensional tensor of wind and ray volume data.

        Returns
        -------
        torch.Tensor
            Result of applying the neural network with skip connections.

        """
        
        output = stacked
        for block in self.blocks[:-1]:
            output = stacked + block(output)

        return self.blocks[-1](output)

    @classmethod
    def _preprocess(cls, u: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Replace the coarse ray volume properties with the logarithms of their
        absolute values, and flip the wind profiles as necessary.

        Parameters
        ----------
        u
            Zonal wind profiles, as passed to `forward`.
        Y
            Coarse ray volume properties, as passed to `forward`.

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
        Return the indices where the input data has nonzero standard deviation.
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
    `Surrogate` accepts a zonal wind profile along with a (coarse) ray volume
    and predicts the time-mean momentum flux profile associated with that ray
    volume over the integration period.
    """

    def forward(self, u: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        After applying the neural network layers, applies the last activation
        function chosen at initialization, and then chooses the appropriate sign
        for each profile based on the zonal wavenumber of each sample. When
        making inferences, the predictions are clamped to be between zero and
        one, so that when multipled by the scaling factor they both are
        sign-definite and respect momentum conservation.
        """

        stacked = self._preprocess(u, Y)
        stacked = self._standardize(stacked)
        signs = torch.sign(Y[:, 2])[:, None]
        output = self._predict(stacked)

        if not self.training:
            output = torch.clamp(output, min=0, max=1)

        return signs * output
    
    def _get_last_layer(self) -> nn.Module:
        """
        `Surrogate` does not have an activation on the last layer. The network
        is allowed to produce negative values during training.
        """

        return nn.Identity()

    @property
    def _n_outputs(self) -> int:
        """
        `Surrogate` has an output for each vertical grid point. Since the fluxes
        are predicted on the cell faces, we can just use `config.n_grid`.
        """

        return config.n_grid

# class Tinkerer(SourceNet):
#     """
#     `Tinkerer` accepts a zonal wind profile along with a (coarse) ray volume
#     and predicts scale factors used to modify some of the ray propertes in
#     order to better match the online behavior of the underlying fine volumes.
#     """

#     props_out = ['k', 'm', 'dm', 'dens']
#     idx_out = [PROP_NAMES.index(prop) for prop in props_out]
    
#     def forward(self, u: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
#         """
#         Take the neural network outputs and use them to scale the appropriate
#         properties of the coarse ray volumes. Then, adjust the density to ensure
#         that momentum flux is conserved.
#         """

#         fluxes = self._get_fluxes(Y)
#         output = self._predict(u, Y).T
#         Y = put(Y.T, self.idx_out, output * Y.T[self.idx_out]).T

#         if params.conservative:
#             factor = fluxes / self._get_fluxes(Y)
#             Y = put(Y.T, 8, Y.T[8] * factor).T

#         return Y

#     @staticmethod
#     def _get_fluxes(Y: torch.Tensor) -> torch.Tensor:
#         """
#         Calculate the momentum flux associated with each coarse ray volume.

#         Parameters
#         ----------
#         Y
#             Tensor of unstandardized coarse ray data, as passed to `forward`,
#             whose second dimension ranges over individual properties.

#         Returns
#         -------
#         torch.Tensor
#             Momentum flux for each ray volume.

#         """

#         *_, k, l, m, dk, dl, dm, dens = Y.T
#         return k * (dens * dk * dl * dm) * cg_r(k, l, m)

#     def _get_last_layer(self) -> nn.Module:
#         """
#         Because `Tinkerer` outputs must be strictly positive, this subclass
#         always uses a Softplus activation as the last layer.
#         """

#         return nn.Softplus()
    
#     @property
#     def _n_outputs(self) -> int:
#         """
#         `Tinkerer` returns a scale factor for each ray property it can modify.
#         """

#         return len(self.props_out)