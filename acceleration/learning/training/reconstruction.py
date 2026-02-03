import numba as nb
import numpy as np
import torch

from ..propagators import EulerianPropagator

def cg_from_T_hat(
    N: torch.Tensor,
    f: torch.Tensor,
    cpt: torch.Tensor,
    T_hat: torch.Tensor
) -> torch.Tensor:
    """
    
    """

    omega_hat = 2 * torch.pi / T_hat
    omega_hat = torch.clip(omega_hat, f + 1e-8)
    wvn_hor = (omega_hat - f) / cpt

    wvn_hor_sq = wvn_hor ** 2
    omega_hat_sq = omega_hat ** 2
    m_sq = wvn_hor_sq * (N ** 2 - omega_hat_sq) / (omega_hat_sq - f ** 2)

    cg = (m_sq ** 0.5) * (
        (omega_hat_sq - f ** 2) /
        (omega_hat * (wvn_hor_sq + m_sq))
    )

    return torch.where(omega_hat > f, cg, 0)

def invert_cg(
    N: np.ndarray,
    f: np.ndarray,
    cg: np.ndarray,
    n_iters: int=500,
    patience: int=30,
    tol: float=5e-4
) -> tuple[np.ndarray, np.ndarray]:
    """
    
    """

    N = torch.as_tensor(N)[:, None]
    f = torch.as_tensor(f)[:, None, None]

    edges = EulerianPropagator._allocate_bins(cg.shape[1], 0.9)
    cpt = torch.as_tensor((edges[:-1] + edges[1:]) / 2)[:, None]

    mask = cg > 1e-14
    cg[~mask] = np.nan

    scales = np.nanstd(cg, (0, -1))[..., None]
    scales[np.isnan(scales)] = 1
    cg[~mask] = 0

    mask = torch.as_tensor(mask).float()
    scales = torch.as_tensor(scales)
    cg = torch.as_tensor(cg)

    T_min, T_max = 2 * torch.pi / N, 2 * torch.pi / f
    logits = torch.zeros_like(cg, requires_grad=True)
    optimizer = torch.optim.Adam([logits], lr=1)

    best_T_hat, best_keep = None, None
    best_loss = torch.inf
    waited = 0

    for n in range(1, n_iters + 1):
        optimizer.zero_grad()

        T_hat = T_min + (T_max - T_min) * torch.sigmoid(logits)
        cg_hat = cg_from_T_hat(N, f, cpt, T_hat)

        losses = (mask * ((cg - cg_hat) / scales) ** 2)
        keep = losses.mean(dim=(1, 2)) < 4 * tol
        loss = losses.mean()

        print(f'Iteration {n}: loss = {loss.item():.6f}')

        if loss < best_loss - 0.1 * tol:
            best_T_hat = T_hat.detach()
            best_keep = keep.detach()
            best_loss = loss
            waited = 0

        else:
            waited = waited + 1
            if waited == patience:
                print('Stopping early due to lack of improvement.')
                break

        if loss < tol:
            print('Tolerance achieved, stopping early.')
            break

        loss.backward()
        optimizer.step()

    best_T_hat = mask * best_T_hat + (1 - mask) * T_max
    return best_T_hat.numpy(), best_keep.numpy()

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
