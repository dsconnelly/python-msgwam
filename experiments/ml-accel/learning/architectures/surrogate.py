import torch

from msgwam import config

from .. import hyperparameters as hp
from .base import SourceNet

class Surrogate(SourceNet):
    """
    A `Surrogate` accepts a zonal wind profile along with a set of ray volume
    properties and predicts the time-mean nondimensional momentum flux profile
    associated with the corresponding packet over the integration period
    """

    def __init__(self) -> None:
        """
        Depending on the hyperparameters, `Surrogate` models may need a softplus
        layer to constrain outputs during postprocessing.
        """

        super().__init__()

        if hp.constrained:
            self._softplus = torch.nn.Softplus()

    @property
    def _n_outputs(self) -> int:
        """
        `Surrogate` has an output for each vertical grid point. The fluxes are
        predicted on the cell faces, so `config.n_grid` is returned.
        """

        return config.n_grid
    
    def _postprocess(
        self,
        _: torch.Tensor,
        X: torch.Tensor,
        output: torch.Tensor
    ) -> torch.Tensor:
        """
        To enforce monotonicity, the layer outputs are interpreted as decrements
        from one over the course of the profile. During evaluation, the profiles
        are clamped to fall between zero and one, so that when redimensionalized
        they both are sign-definite and respect momentum conservation.
        """

        if hp.constrained:
            output = self._softplus(output)
            output = 1 - torch.cumsum(output, dim=1)

        if not self.training:
            output = torch.clamp(output, min=0, max=1)

        signs = torch.sign(X[:, 0])[:, None]
        return signs * output
