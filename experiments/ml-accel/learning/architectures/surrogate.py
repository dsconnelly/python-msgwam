import torch, torch.nn as nn

from msgwam import config

from .. import hyperparameters as hp
from .base import SourceNet

class Surrogate(SourceNet):
    """
    A `Surrogate` accepts a zonal wind profile along with a set of ray volume
    properties and predicts the time-mean nondimensional momentum flux profile
    associated with the corresponding packet over the integration period. If the
    model is configured to be constrained, it also accepts a height coordinate
    and makes point predictions instead of predicting the whole flux profile.
    """

    def __init__(self) -> None:
        """
        At initialization, a `Surrogate` creates a dummy z array that can be
        used later to evaluate the basis functions. The actual values are less
        important than that they are decreasing.
        """

        super().__init__()

        z = torch.linspace(-3, 3, config.n_grid)
        self._z = -0.9 * z[None, None]
        self._z_max = z.max()

    @property
    def _n_final(self) -> int:
        """
        If the `Surrogate` is not constrained to be monotonic, then the output
        of the last block is the neural network output, and so it should have
        one value for each vertical grid point. If the model is constrained,
        then the last layer provides amplitude, shape, and shift parameters to
        be passed to the basis functions.
        """

        return 3 * hp.n_basis if hp.n_basis > 0 else config.n_grid
    
    def _postprocess(self, _, output: torch.Tensor) -> torch.Tensor:
        """
        At inference time, outputs are clamepd to fall between zero and one, so
        that they both are sign-definite and respect momentum conservation.
        """

        if hp.n_basis == 0:
            if not self.training:
                output = torch.clamp(output, min=0, max=1)

            return output

        output = output.reshape(-1, 3, hp.n_basis, 1)
        amp, shape, shift = output.transpose(0, 1)

        amp = torch.softmax(amp, dim=1)
        shape = nn.functional.softplus(shape)
        shift = torch.tanh(shift) * self._z_max
        
        arg = shape * (self._z - shift)
        curves = amp * self._basis_func(arg)

        return curves.sum(dim=1)
    
    @staticmethod
    def _basis_func(z: torch.Tensor) -> torch.Tensor:
        """
        Compute the normalized version of the basis function, which must have
        unit slope at the origin and be bounded between zero and one.

        Parameters
        ----------
        z
            Tensor of input values.

        Returns
        -------
        torch.Tensor
            Basis function values.

        """

        if hp.basis_func == 'logistic':
            return 1 / (1 + torch.exp(-4 * z))
        
        if hp.basis_func == 'quadratic':
            return (1 + 2 * z / torch.sqrt(1 + (2 * z) ** 2)) / 2
        
        if hp.basis_func == 'tanh':
            return (1 + torch.tanh(2 * z)) / 2

        raise ValueError(f'Unknown value for basis_func: {hp.basis_func}')
