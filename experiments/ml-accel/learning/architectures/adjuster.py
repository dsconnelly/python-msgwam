import torch

from .base import SourceNet

class Adjuster(SourceNet):
    """
    An `Adjuster` accepts information about the zonal wind and the coarse ray
    volumes to be launched, and returns new zonal and vertical wavenumbers such
    that the adjusted ray volumes behave more like the underlying fine rays.
    """

    @property
    def _n_outputs(self) -> int:
        """
        The `Adjuster` has two outputs, for k and m.
        """

        return 2
    
    def _postprocess(self, output: torch.Tensor) -> torch.Tensor:
        """At present, `Adjuster` outputs are not postprocessed."""

        return output