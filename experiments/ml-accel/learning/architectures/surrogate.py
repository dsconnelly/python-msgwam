import torch, torch.nn as nn

from msgwam import config

from .. import hyperparameters as hp
from ..utils import get_proxy_statistics, transform_proxies

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

    def init_stats(self, *Xs):
        """
        
        """

        if hp.basis_type == 'none':
            return super().init_stats(*Xs)
        
        super().init_stats(Xs[0])
        means, stds = get_proxy_statistics(Xs[1])

        self.means.append(means)
        self.stds.append(stds)

    @property
    def _n_final(self) -> int:
        """
        If the `Surrogate` is not constrained to be monotonic, then the output
        of the last block is the neural network output, and so it should have
        one value for each vertical grid point. If the model is constrained,
        then the last layer provides amplitude, shape, and shift parameters to
        be passed to the basis functions.
        """

        return config.n_grid if hp.basis_type == 'none' else 3 * hp.n_basis

    def _postprocess(self, _, output: torch.Tensor) -> torch.Tensor:
        """
        At inference time, if the model is predicting flux profiles directly,
        outputs are clamped to fall between zero and one, so that they both are
        sign-definite and respect momentum conservation.
        """

        if (hp.basis_type == 'none') and (not self.training):
            output = torch.clamp(output, min=0, max=1)

        if hp.basis_type != 'none':
            alpha = 0.01 if self.training else 0
            output = output.reshape(-1, 3, hp.n_basis)
            output = transform_proxies(output, alpha=alpha)

        return output
