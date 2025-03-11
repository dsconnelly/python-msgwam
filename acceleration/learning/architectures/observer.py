import torch, torch.nn as nn

from msgwam import config

from ...hyperparameters import architectures as hp
from .base import BaseNet
from .utils import apply_with_skips, get_block

class Observer(BaseNet):
    """
    An `Observer` takes the latent representation of the ray volume state and
    estimates the corresponding (signed) momentum flux profile.
    """

    def _forward(self, Z: torch.Tensor) -> torch.Tensor:
        """
        The `Observer` may have skip connections, where the latent space data is
        added to the intermediate outputs.
        """

        return apply_with_skips(self._blocks, Z)

    def _init_layers(self):
        """
        Create a set of fully-connected blocks, ending with a layer with the
        appropriate dimension to be interpreted as a flux profile.
        """

        blocks = []
        for i in range(hp.n_blocks):
            final = i == hp.n_blocks - 1
            n_out = config.n_grid if final else hp.n_latent
            sizes = [hp.n_latent] * (hp.n_per_block - 1) + [n_out]
            blocks.append(get_block(sizes, final))

        self._blocks = nn.ModuleList(blocks)

    @property
    def _inputs(self) -> list[str]:
        """The `Observer` makes predictions from the latent space."""
        return ['Z']
    
    @property
    def _output(self) -> str:
        """The `Observer` outputs flux profiles."""
        return 'F'