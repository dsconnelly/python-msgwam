import torch, torch.nn as nn

from msgwam import config

from ... import hyperparameters as hp
from ..utils import init_proxies, transform_proxies

from .base import SourceNet

class Surrogate(SourceNet):
    """
    A `Surrogate` accepts a zonal wind profile along with a set of ray volume
    properties and predicts the time-mean nondimensional momentum flux profile
    associated with the corresponding packet over the integration period. If the
    model is configured to be constrained, it makes this prediction by returning
    parameters to be passed to a family of basis functions satisfying certain
    properties, which are then used to compute the profile.
    """

    def __init__(self):
        """
        
        """

        super().__init__()
        
        if hp.architectures.basis_type != 'none':
            guess = init_proxies(1).flatten()
            layer: nn.Linear = self._blocks[-1][-1]

            with torch.no_grad():
                nn.init.zeros_(layer.weight)
                layer.bias.data.copy_(guess)

            self._alpha = 0.01
            n_rolloff = hp.training.rolloff_end - hp.training.rolloff_start
            self._decrement = self.alpha / n_rolloff

    def step(self, n_epoch: int):
        """
        If this `Surrogate` is constrained, then the negative slope used to
        transform proxy amplitudes is gradually zeroed out during training.

        Parameters
        ----------
        n_epoch
            Current epoch.

        """

        if hp.architectures.basis_type != 'none':
            if hp.training.rolloff_start <= n_epoch <= hp.training.rolloff_end:
                self._alpha = max(self._alpha - self._decrement, 0)

    @property
    def alpha(self) -> float:
        """
        Get the negative slope that should be used in transforming the proxy
        amplitudes, if the network is constrained.
        """

        if self.training and (hp.architectures.basis_type != 'none'):
            return self._alpha
        
        return 0

    @property
    def _n_final(self) -> int:
        """
        If the `Surrogate` is not constrained to be monotonic, then the output
        of the last block is the neural network output, and so it should have
        one value for each vertical grid point. If the model is constrained,
        then the last layer provides amplitude, shape, and shift parameters to
        be passed to the basis functions.
        """

        if hp.architectures.basis_type == 'none':
            return config.n_grid
        
        return 3 * hp.architectures.n_basis

    def _postprocess(self, _, output: torch.Tensor) -> torch.Tensor:
        """
        At inference time, if the model is predicting flux profiles directly,
        outputs are clamped to fall between zero and one, so that they both are
        sign-definite and respect momentum conservation.
        """

        if (hp.architectures.basis_type == 'none') and (not self.training):
            output = torch.clamp(output, min=0, max=1)

        if hp.architectures.basis_type != 'none':
            output = output.reshape(-1, 3, hp.architectures.n_basis)
            output = transform_proxies(output, alpha=self.alpha)

        return output
