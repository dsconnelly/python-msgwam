import torch, torch.nn as nn

from msgwam import config

from .base import SourceNet

class Surrogate(SourceNet):
    """
    A `Surrogate` accepts information about the zonal wind and the source ray
    volume properties, and predicts the time-mean nondimensional momentum flux
    profile associated with the corresponding packet over the integration.
    """

    def __init__(self) -> None:
        """
        
        """

        super().__init__()
        layer: nn.Linear = self._shared[-1]
        guess = 0.5 * torch.ones(self._n_outputs)

        with torch.no_grad():
            nn.init.zeros_(layer.weight)
            layer.bias.data.copy_(guess)

    @property
    def _n_outputs(self) -> int:
        """
        The `Surrogate` predicts a flux at each point in the vertical grid.
        """

        return config.n_grid
    
    def _postprocess(self, output: torch.Tensor) -> torch.Tensor:
        """
        If the `Surrogate` is in evaluation mode, then the predicted fluxes are
        constrained to be in [0, 1] (in nondimensional terms) and decreasing as
        a function of height.
        """
        
        if not self.training:
            output = torch.clamp(output, min=0, max=1)
            output = torch.cummin(output, dim=1)[0]

        return output