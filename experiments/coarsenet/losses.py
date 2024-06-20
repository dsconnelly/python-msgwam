import sys
sys.path.insert(0, '.')

import torch, torch.nn as nn

from msgwam.mean import MeanState

from architectures import CoarseNet
from utils import integrate_batches
from preprocessing import SMOOTHING

class RegularizedMSELoss(nn.Module):    
    def __init__(self, z_max: float=35e3) -> None:
        """
        Initialize a module to compute MSE and regularization losses.

        Parameters
        ----------
        z_max
            Level above which to ignore errors in momentum flux.

        """

        super().__init__()
        centers = MeanState().z_centers
        self.keep = centers <= z_max

    def forward(
        self,
        u: torch.Tensor,
        Y: torch.Tensor,
        Z: torch.Tensor,
        model: CoarseNet
    ) -> torch.Tensor:
        """
        Calculate the MSE loss averaged over the specified vertical levels.

        Parameters
        ----------
        u
            Zonal wind profile for the given batch.
        Y
            Coarse ray volumes for the given batch.
        Z
            Time-averaged momentum flux profiles for the given batch.
        model
            `CoarseNet` instance being trained.

        Returns
        -------
        torch.Tensor
            Squared error between the target momentum flux profile and the
            profile obtained by integrating with the replacement ray volume.
            Both profiles are normalized by the absolute maximum of the target
            profile before the error is computed.

        """

        v = torch.zeros_like(u)
        wind = torch.stack((u, v), dim=1)
        
        output = model(u[0], Y)
        spectrum = model.build_spectrum(Y, output)

        Z = Z[..., self.keep]
        Z_hat = integrate_batches(
            wind, spectrum,
            rays_per_packet=1,
            smoothing=SMOOTHING
        ).mean(dim=1)[..., self.keep]

        scales, _ = torch.abs(Z).max(dim=-1)
        sdx = scales > 0

        errors = torch.zeros_like(Z)
        errors[sdx] = (Z - Z_hat)[sdx] / scales[sdx, None]

        return (errors ** 2).mean()
