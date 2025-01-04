from __future__ import annotations
import torch, torch.nn as nn

from msgwam import config

from ... import hyperparameters as hp
from ..utils import apply_basis, init_proxies, transform_proxies

from .base import SourceNet

class Surrogate(SourceNet):
    """
    A `Surrogate` accepts information about the zonal wind and the source ray
    volume properties, and predicts the time-mean nondimensional momentum flux
    profile associated with the corresponding packet over the integration. This
    class can behave in various ways, depending on the constraint strategy set
    in the hyperparameters.
    """

    def __init__(self) -> None:
        """
        If the model is constrained, we initialize the last bias vector to give
        reasonable guesses, in the hopes that this speeds convergence.
        """

        super().__init__()
        if hp.architectures.constraint == 'none':
            return
        
        if hp.architectures.constraint == 'increment':
            guess = torch.log(torch.ones(self._n_final) / self._n_final)

        elif hp.architectures.constraint == 'logistic':
            guess = init_proxies(1).flatten()

        with torch.no_grad():
            layer: nn.Linear = self._blocks[-1][-1]
            nn.init.zeros_(layer.weight)
            layer.bias.data.copy_(guess)

    @property
    def _n_final(self) -> int:
        """
        If the `Surrogate` is predicting the fluxes directly, it needs one
        output for each point in the vertical grid. If it is predicting the
        increments, it needs one fewer point than that, and if it is predicting
        fluxes by means of basis functions, it needs three outputs for each
        function in the expansion.
        """

        if hp.architectures.constraint == 'logistic':
            return 3 * hp.architectures.n_basis
        
        return config.n_grid

    def _postprocess(self, _, output: torch.Tensor) -> torch.Tensor:
        """
        This function ensures that the `Surrogate` returns an (unsigned) flux
        profile for each sample. For unconstrained models, this is as simple as
        clamping the data in evaluation mode, but for constrained models here is
        where the constraint strategies are applied.
        """

        if hp.architectures.constraint == 'none' and not self.training:
            output = torch.clamp(output, min=0, max=1)

        elif hp.architectures.constraint == 'increment':
            output = torch.exp(output)
            norms = output.sum(dim=1, keepdim=True)
            norms = torch.clamp(norms, min=1)

            output = torch.cumsum(output / norms, dim=1)
            output = torch.flip(output, dims=(1,))

        elif hp.architectures.constraint == 'logistic':
            output = output.reshape(-1, 3, hp.architectures.n_basis)
            output = transform_proxies(output, 1e-3 if self.training else 0)
            output = apply_basis(output)

        return output