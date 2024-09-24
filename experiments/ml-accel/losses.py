from __future__ import annotations
from typing import TYPE_CHECKING, Callable, Optional

import torch, torch.nn as nn

if TYPE_CHECKING:
    Surrogate = Callable[..., torch.Tensor]

class ProfileLoss(nn.Module):
    """
    Module to compute losses between momentum flux profiles. Flexible enough to
    handle both `SurrogateNet` and `CoarseNet` training.
    """

    def __init__(self, surrogate: Optional[Surrogate]=None) -> None:
        """
        Save the surrogate, if there is one.

        Parameters
        ----------
        surrogate
            Map from zonal wind profiles and coarse ray volume properties to
            momentum flux profiles, if training a `CoarseNet`, otherwise `None`.

        """

        super().__init__()
        self.surrogate = surrogate

    def forward(
        self,
        u: torch.Tensor,
        output: torch.Tensor,
        Z: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate mean squared error between (nondimensionalized) momentum flux
        profiles, first propagating with the surrogate if necessary.

        Parameters
        ----------
        u
            Two-dimensional tensor of zonal wind profiles.
        output
            Neural network output. If `self.surrogate` is not `None`, these will
            be interpreted as coarse ray volume properties and converted to flux
            profiles with the provided mapping. Otherwise, these will be treated
            as the momentum flux profiles themselves and compared directly.
        Z
            Two-dimensional tensor of true momentum flux profiles.

        Returns
        -------
        torch.Tensor
            Mean squared error across all samples and vertical levels.
 
        """

        if self.surrogate is not None:
            output = self.surrogate(u, output)

        return ((output - Z) ** 2).mean()
