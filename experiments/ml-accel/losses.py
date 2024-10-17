import torch, torch.nn as nn

from msgwam.mean import MeanState

class DivergenceLoss(nn.Module):
    """
    Loss function to compute losses between flux divergence profiles using the
    Earth mover's distance.
    """

    def __init__(self) -> None:
        """
        `DivergenceLoss` ignores drag below a certain level, to avoid spurious
        dependence on the launch level. At initialization, the module stores a
        boolean tensor used to index only drags above that level.
        """

        super().__init__()
        grid = MeanState().z_centers
        self.keep = grid > 20e3

    def forward(self, Z: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """
        Calculate the flux divergence of each profile, and then return the loss
        as the sum of the Earth mover's distance over all profiles.

        Parameters
        ----------
        Z
            Two-dimensional tensor of target (nondimensional) flux profiles.
        output
            Two-dimensional tensor of output (nondimensional) flux profiles.
        
        Returns
        -------
        torch.Tensor
            Total Earth mover's distance.

        """

        # Z = torch.diff(Z, dim=1)[:, self.keep]
        # output = torch.diff(output, dim=1)[:, self.keep]

        cdf_1 = self._get_cdf(Z)
        cdf_2 = self._get_cdf(output)

        return abs(cdf_1 - cdf_2).mean()
    
    def _get_cdf(self, a: torch.Tensor) -> torch.Tensor:
        """
        
        """

        scale = a.sum(dim=1)
        scale[scale == 0] = 1
        cdf = torch.cumsum(a, dim=1)

        return cdf / scale[:, None]
