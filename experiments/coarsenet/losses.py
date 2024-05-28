import sys
sys.path.insert(0, '.')

import torch, torch.nn as nn

from msgwam.mean import MeanState

from architectures import CoarseNet
from utils import integrate_batches

class RegularizedMSELoss(nn.Module):

    def __init__(self, z_min: float=15e3, z_max: float=50e3) -> None:
        """
        Initialize a module to compute MSE and regularization losses.

        Parameters
        ----------
        z_min, z_max
            Minimum and maximum heights bounding the vertical region on which
            the MSE loss should be computed.

        """

        super().__init__()
        centers = MeanState().z_centers
        self.keep = (z_min < centers) & (centers < z_max)

    def forward(
        self,
        u: torch.Tensor,
        X: torch.Tensor,
        Z: torch.Tensor,
        model: CoarseNet,
        smoothing: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the regularization and MSE losses.

        Parameters
        ----------
        u
            Zonal wind profile for the given batch.
        X
            Wave packets for the given batch.
        Z
            Time-averaged momentum flux profiles for the given batch.
        model
            `CoarseNet` instance being trained.
        smoothing
            Smoothing value to use in Gaussian projection.

        Returns
        -------
        torch.Tensor
            Regularization loss, penalizing squared deviations from the default
            replacement ray volumes.
        torch.Tensor
            Squared error between the target momentum flux profile and the
            profile obtained by integrating with the replacement ray volume.
            Both profiles are normalized by the absolute maximum of the target
            profile before the error is computed.

        """

        output = model(u, X)
        reg = ((output - 1) ** 2).mean()
        spectrum = model.build_spectrum(X, output)
        wind = torch.vstack((u, torch.zeros_like(u)))

        Z_hat = integrate_batches(
            wind, spectrum,
            rays_per_packet=1,
            smoothing=smoothing
        ).mean(dim=1)[..., self.keep]

        scales, _ = abs(Z).max(dim=-1)
        errors = (Z_hat - Z) / scales[:, None]
        mse = (errors ** 2).mean()

        return reg, mse
