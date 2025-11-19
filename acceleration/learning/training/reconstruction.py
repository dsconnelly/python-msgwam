from itertools import product

import numba as nb
import numpy as np
import tqdm

@nb.njit
def get_vertical_flux(
    M: np.ndarray,
    dF: np.ndarray,
    F_est: np.ndarray
) -> np.ndarray:
    """
    Get the vertical flux given the flux divergence, assuming that the flux is
    proportional to the momentum concentration unless there is no pre-existing
    momentum in a row and flux needs to be exported upwards.

    Parameters
    ----------
    M
        Input flux at each height and phase speed bin.
    dF
        Flux divergence at each height and phase speed bin.
    F_est
        Estimate of the vertical momentum flux to use when there is no bulk
        momentum (e.g. from projections of the online flux).

    Returns
    -------
    np.ndarray
        Array of vertical fluxes. Has one more vertical dimension than `M`.

    """

    n_samples, n_bins, n_vert = M.shape
    F = np.zeros((n_samples, n_bins, n_vert + 1))

    for i in range(n_samples):
        for k in range(n_vert):
            delta = F[i, :, k] - dF[i, :, k]
            start = 0

            while start < n_bins:
                if np.abs(delta[start]) < 1e-14:
                    start = start + 1
                    continue

                end = start + 1
                while np.abs(delta[end]) > 1e-14 and end < n_bins:
                    end = end + 1

                sdx = slice(start, end)
                F_out = delta[sdx].sum()
                avail = M[i, sdx, k].sum()
                est = F_est[i, sdx, k + 1].sum()

                if avail > 1e-14:
                    F[i, sdx, k + 1] = F_out * M[i, sdx, k] / avail

                elif est > 1e-14:
                    F[i, sdx, k + 1] = F_out * F_est[i, sdx, k + 1] / est

                else:
                    F[i, sdx, k + 1] = F_out / (end - start)

                start = end
            
    a = F.sum(axis=1)[:, None]
    a = np.where(a > 0, a, 1)
    
    F = F * (F > 1e-14)
    b = F.sum(axis=1)[:, None]
    b = np.where(b > 0, b, 1)

    return a * F / b

@nb.njit
def get_horizontal_flux(F_v: np.ndarray) -> np.ndarray:
    """
    Given the vertical flux, reconstruct the flux between phase speed bins,
    assuming the flux field is irrotational. Most useful if the vertical flux is
    available but the potential is not (e.g. if the vertical flux is actually a
    prediction by the neural network).

    Parameters
    ----------
    F_v
        Vertical flux, including padding in the vertical.

    Returns
    -------
    F_h
        Horizontal flux, including padding in the phase speed dimension.

    """

    n_samples, n_bins, n_faces = F_v.shape
    F_h = np.zeros((n_samples, n_bins + 1, n_faces - 1))

    for i in range(n_samples):
        for k in range(1, n_faces - 1):
            F_h[i, 1:-1, k] = F_v[i, 1:, k] - F_v[i, :-1, k]
            F_h[i, 1:-1, k] += F_h[i, 1:-1, k - 1]

    return F_h

def get_potential(dF: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Given the change in wave momentum not associated with the various sinks we
    solve a Poisson equation to recover a potential whose gradients are the
    corresponding fluxes.

    Parameters
    ----------
    dF
        Change in momentum in each cell minus the dissipative sink.
    
    Returns
    -------
    np.ndarray
        Value of the potential function at each cell center.
    np.ndarray
        Boolean array indicating whether each linear solve succeeded.

    """

    n_samples, n_bins, n_z = dF.shape
    n_equations = n_bins * n_z

    A = np.zeros((n_equations, n_equations))
    vdx = np.arange(n_equations).reshape(n_bins, n_z)

    for n, (j, k) in enumerate(product(range(n_bins), range(n_z))):
        A[n, vdx[max(0, j - 1), k]] += 1
        A[n, vdx[j, max(0, k - 1)]] += 1

        A[n, vdx[min(n_bins - 1, j + 1), k]] += 1
        A[n, vdx[j, min(n_z - 1, k + 1)]] += 1

        A[n, vdx[j, k]] -= 4
    
    phi = np.zeros_like(dF)
    success = np.ones(n_samples).astype(bool)

    for i in tqdm.trange(n_samples):
        try:
            res = np.linalg.solve(A, -dF[i].flatten())
            phi[i] = res.reshape(n_bins, n_z)

        except np.linalg.LinAlgError:
            success[i] = False

    phi = phi - phi.min((1, 2), keepdims=True)
    phi = np.maximum.accumulate(phi, axis=2)

    return phi, success