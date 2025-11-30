import numba as nb
import numpy as np
import torch

@nb.njit
def correct_bins(
    M: np.ndarray,
    u_old: np.ndarray,
    u_new: np.ndarray,
    edges: np.ndarray,
    conservative: bool=True
) -> tuple[np.ndarray, np.ndarray]:
    """
    Correct a set of bulk momentum density profiles to account for the fact that
    the mean wind, and so the definition of the bins, has shifted.
    """

    for i in range(M.shape[0]):
        for k in range(M.shape[2]):
            edges_old = edges + u_old[i, k]
            edges_new = edges + u_new[i, k]
            
            A = get_A(edges_old, edges_new, conservative)
            M[i, :, k] = A @ M[i, :, k]

    return M

@nb.njit
def get_A(
    edges_old: np.ndarray,
    edges_new: np.ndarray,
    conservative: bool
) -> np.ndarray:
    """
    Get a matrix that gives the percentage of each old bin that should be
    transferred to each new bin.
    """

    n_bins = len(edges_old) - 1
    A = np.zeros((n_bins, n_bins))

    edges_new[-1] = max(edges_old[-1], edges_new[-1])
    if conservative:
        edges_new[0] = min(edges_old[0], edges_new[0])

    pairs_old = (edges_old[:-1], edges_old[1:])
    pairs_new = (edges_new[:-1], edges_new[1:])

    for j, (a_old, b_old) in enumerate(zip(*pairs_old)):
        for i, (a_new, b_new) in enumerate(zip(*pairs_new)):
            a = max(a_old, a_new)
            b = min(b_old, b_new)

            if a >= b:
                continue

            frac = (b - a) / (b_old - a_old)
            A[i, j] = frac

    return A

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
