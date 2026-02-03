from typing import Literal, Optional

import numba as nb
import numpy as np
import torch

from ...hyperparameters import generation as hp

_Array = np.ndarray | torch.Tensor

@nb.njit
def correct_bins(
    data: np.ndarray,
    edges_old: np.ndarray,
    edges_new: np.ndarray,
    conservative: bool
) -> tuple[np.ndarray, np.ndarray]:
    """
    
    """

    n_bins = len(data)
    fracs = np.zeros(n_bins)
    out = np.zeros(n_bins)

    edges_new[-1] = max(edges_old[-1], edges_new[-1])
    if conservative:
        edges_new[0] = min(edges_old[0], edges_new[0])

    i, j = 0, 0
    while i < n_bins:

        a_old, b_old = edges_old[i:(i + 2)]
        a_new, b_new = edges_new[j:(j + 2)]

        a = max(a_old, a_new)
        b = min(b_old, b_new)

        frac = (b - a) / (b_old - a_old)
        frac = max(0, min(1, frac))

        fracs[i] = fracs[i] + frac
        out[j] = out[j] + frac * data[i]

        if b_old > b_new:
            j = j + 1
        else:
            i = i + 1

    return fracs, out

@nb.njit
def get_A(
    edges_old: np.ndarray,
    edges_new: np.ndarray,
    conservative: bool
) -> np.ndarray:
    """
    Get a matrix that gives the transfer from one set of phase speed bins to
    another, useful for adjusting between levels with different mean winds or
    between time steps.

    Parameters
    ----------
    edges_old, edges_new
        Edges of the old and new phase speed bins.
    conservative
        Whether momentum that is below the lowest new phase speed bin should be
        lost (if `False`) or shifted into the new lowest bin (if `True`).
    
    Returns
    -------
    np.ndarray
        Transfer matrix such that `mom_new = A @ mom_old`.

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
            if b_new < a_old:
                continue

            if b_old < a_new:
                break

            a = max(a_old, a_new)
            b = min(b_old, b_new)

            frac = (b - a) / (b_old - a_old)
            A[i, j] = frac

    return A

def get_bin_edges(
    n_bins: Optional[int]=None,
    mode: Literal['coarsen', 'from_left']='from_left'
) -> np.ndarray:
    """
    Get phase speed bin edges.

    Parameters
    ----------
    n_bins
        How many bins to keep. If `None`, uses the value from the currently
        loaded static hyperparameter file.
    mode
        How to reduce the number of bins, if `n_bins is not None`. Can be either
        `'coarsen'`, in which case the fine bins will be grouped evenly, or
        `'from_left'`, in which case the first `n_bins - 1` will be left as is
        and the last bin will contain all the remaining fine bins.

    Returns
    -------
    np.ndarray
        Array of bin edges.

    """

    edges = np.linspace(0, 100, hp.n_bins + 1)
    if n_bins is None:
        return edges

    if mode == 'coarsen':
        if n_bins % hp.n_bins:
            raise ValueError(f'n_bins must divide {hp.n_bins}')

        left = edges[:-1].reshape(n_bins, -1)[:, 0]
        return np.concatenate((left, edges[-1:]))

    elif mode == 'from_left':
        return np.concatenate((edges[:n_bins], [edges[-1]]))

    raise ValueError(f'Unknown mode {mode}')

def get_cg_r(
    wvn_hor: _Array,
    cp_tilde: _Array,
    N: _Array,
    f: _Array
) -> _Array:
    """
    Calculate the group velocity from the wavenumber and Coriolis-adjusted
    intrinsic phase speed.
    """

    omega_hat = recover_omega_hat(wvn_hor, cp_tilde, N, f)
    omega_hat_sq = omega_hat ** 2
    wvn_hor_sq = wvn_hor ** 2

    m_sq = wvn_hor_sq * (N ** 2 - omega_hat_sq) / (omega_hat_sq - f ** 2)

    return (m_sq ** 0.5) * (
        (omega_hat_sq - f ** 2) /
        (omega_hat * (wvn_hor_sq + m_sq))
    )

def recover_omega_hat(
    wvn_hor: _Array,
    cp_tilde: _Array,
    N: _Array,
    f: _Array
) -> _Array:
    """
    Recover the intrinsic frequency from the wavenumber and Coriolis-adjusted
    intrinsic frequency, taking care to make sure it is bounded by the Coriolis
    and buoyancy frequencies.
    """

    lib = torch if isinstance(wvn_hor, torch.Tensor) else np
    omega_hat = wvn_hor * cp_tilde + lib.abs(f)
    
    return lib.clip(omega_hat, lib.abs(f) + 1e-8, N)

def get_qdx(k: np.ndarray, l: np.ndarray) -> np.ndarray:
    """
    Get an index array indicating which wavenumber quadrant each ray volume
    should be projected into.

    Parameters
    ----------
    k, l
        Zonal and meridional wavenumbers of each wavenumber quadrant.
    
    Returns
    -------
    np.ndarray
        Index array where integers correspond to the k > 0, l > 0, k < 0, and
        l < 0 quadrants, respectively. Indices corresponding to inactive columns
        in the state array will be -1.

    """

    return ((k > 0) + 2 * (l > 0) + 3 * (k < 0) + 4 * (l < 0)) - 1

@nb.njit
def get_transports(
    M: np.ndarray,
    cg: np.ndarray,
    edges: np.ndarray,
    wind: np.ndarray,
    dt_o_dz: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the transport of momentum between vertical cells according to the
    group velocity, taking into account both non-negativity constraints and the
    change in bin edges between vertical levels in the presence of shear.
    """

    _, n_wvn, n_cp_tilde, n_z = M.shape
    phi_in = np.zeros((4, n_wvn, n_cp_tilde, n_z))
    phi_out = np.zeros((4, n_wvn, n_cp_tilde, n_z))

    for q in range(4):
        for i in range(n_wvn):
            for k in range(n_z):
                avail = M[q, i, :, k] + phi_in[q, i, :, k]
                transport = cg[q, i, :, k] * M[q, i, :, k] * dt_o_dz
                phi_out[q, i, :, k] = np.minimum(avail, transport)

                if k < n_z - 1:
                    edges_old = edges + wind[q, k]
                    edges_new = edges + wind[q, k + 1]

                    fracs, out = correct_bins(
                        phi_out[q, i, :, k],
                        edges_old,
                        edges_new,
                        False
                    )

                    phi_in[q, i, :, k + 1] = out
                    phi_out[q, i, :, k] = fracs * phi_out[q, i, :, k]

    return phi_in, phi_out

@nb.njit
def project(
    data: np.ndarray,
    c_lo: np.ndarray,
    c_hi: np.ndarray,
    r_lo: np.ndarray,
    r_hi: np.ndarray,
    edges_c: np.ndarray,
    edges_z: np.ndarray,
    qdx: np.ndarray,
    out: np.ndarray
) -> np.ndarray:
    """
    Projection operator as used by the ray tracer, but also taking into account
    the location in phase speed space.

    Parameters
    ----------
    data
        Data associated with each ray volume to project. Assumed to be a density
        in physical space but not in spectral space.
    c_lo, c_hi
        Lower and upper bounds of each ray volume in phase speed.
    r_lo, r_hi
        Lower and upper bounds of each ray volume in the vertical.
    edges_z, edges_c
        Edges of grid cells in the vertical and in phase speed.
    qdx
        Index array indicating which output array to project each ray volume
        into. Useful for projecting different wavenumber quadrants separately.
    out
        Where to write projected data.

    Returns
    -------
    np.ndarray
        Array of shape (pdx.max() + 1, len(edges_c) - 1, len(edges_z) - 1) which
        contains the contribution from `data` to the profile in the appropriate
        phase speed bin.

    """

    for i, (v, q) in enumerate(zip(data, qdx)):
        if np.isnan(v) or q < 0:
            continue

        j = 0
        while c_lo[i] > edges_c[j + 1]:
            j = j + 1

        k_start = int((r_lo[i] - edges_z[0]) / (edges_z[1] - edges_z[0]))

        while j < edges_c.shape[0] - 1:
            if edges_c[j] > c_hi[i]:
                break

            frac_c = min(edges_c[j + 1], c_hi[i]) - max(edges_c[j], c_lo[i])
            frac_c = frac_c / (edges_c[j + 1] - edges_c[j])

            k = k_start
            while k < edges_z.shape[0] - 1:
                if edges_z[k] > r_hi[i]:
                    break

                frac_z = min(edges_z[k + 1], r_hi[i]) - max(edges_z[k], r_lo[i])
                frac_z = frac_z / (edges_z[k + 1] - edges_z[k])
                out[q, j, k] += frac_c * frac_z * v
                k = k + 1

            j = j + 1
