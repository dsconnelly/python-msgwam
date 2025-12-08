import numba as nb
import numpy as np
import torch

def get_dM(Y: torch.Tensor) -> torch.Tensor:
    """
    Get the total change in momentum density from the vertical fluxes and sink
    profiles, without correction (for use during training.)
    """

    F, D = Y[:, 0], Y[:, 1]
    F = torch.nn.functional.pad(F, (1, 0))

    return F[..., :-1] - F[..., 1:] + D

@nb.njit
def get_vertical_flux(dF: np.ndarray, F_est: np.ndarray) -> np.ndarray:
    """
    Get the vertical flux using an estimate from the projected fluxes.
    """

    F = np.zeros_like(F_est)
    for i in range(F_est.shape[0]):
        for k in range(F_est.shape[2] - 1):
            F_in = F[i, :, k].sum()
            F_out = F_in - dF[i, k]

            if F_out < 1e-12:
                continue

            q = k
            F_prop = F_est[i, :, k + 1]
            scale = F_prop.sum()

            while scale <= 0 and q > -1:
                F_prop =  F[i, :, q]
                scale = F_prop.sum()
                q = q - 1

            if scale != 0:
                F[i, :, k + 1] = F_out * F_prop / scale

            else:
                F[i, :, k + 1] = F_out / F_est.shape[1]

    return F
