import numba as nb
import numpy as np

from .. import config

@nb.njit
def get_max_intersects(
    r: np.ndarray,
    dr: np.ndarray,
    edges: np.ndarray,
    profile: np.ndarray
) -> np.ndarray:
    """
    Find the maximum value in a vertical profile intersected by each ray.

    Parameters
    ----------
    r
        Positions of ray volume centers.
    dr
        Ray volume extents.
    edges
        Edges of the vertical grid regions where the profile is stored.
    profile
        Vertical profile (e.g. dissipation constants) to search.

    Returns
    -------
    np.ndarray
        Array with `len(r)` elements containing the maximum value for each ray.

    """

    r_lo = r - 0.5 * dr
    r_hi = r + 0.5 * dr

    maxes = np.zeros(len(r))
    for i, (a, b) in enumerate(zip(r_lo, r_hi)):
        if np.isnan(a):
            continue

        for j, (z_lo, z_hi) in enumerate(zip(edges[:-1], edges[1:])):
            if b < z_lo:
                break

            if z_hi < a:
                continue

            if profile[j] > maxes[i]:
                maxes[i] = profile[j]

    return maxes

@nb.njit
def get_steady_action_fluxes(
    k: np.ndarray,
    l: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    N: np.ndarray,
    rho: np.ndarray,
    omega: np.ndarray,
    source_flux: np.ndarray
) -> np.ndarray:
    """
    Propagate the waves launched at the source according to the steady-state
    monochromatic scheme described in Bölöni et al.

    Parameters
    ----------
    k, l
        Arrays of zonal and meridional wavenumbers, respectively.
    u, v
        Arrays of zonal and meridional mean wind, respectively, interpolated to
        vertical grid cell faces.
    N
        Array of buoyancy frequencies at vertical grid cell faces.
    rho
        Array of mean state densities at vertical grid cell faces.
    omega
        Extrinsic frequency of each wave. In the steady-state approximation, it
        is assumed that omega is conserved.
    source_flux
        Group velocity times action associated with each wave at the source. In
        the steady-state approximation, this quantity is conserved except in the
        presence of wave saturation.

    Returns
    -------
    np.ndarray
        Array of shape `(len(u), len(k))` whose entry at [j, i] gives the group
        velocity times action associated with the ith ray at the jth vertical
        grid face. Can be multiplied by the appropriate wavenumber and summed
        over the second dimension to obtain a momentum flux profile.
    
    """

    n_faces, n_waves = len(u), len(k)
    out = np.zeros((n_waves, n_faces))
    out[:, 0] = source_flux

    for i in range(n_waves):
        wvn_hor_sq = k[i] ** 2 + l[i] ** 2

        for j in range(1, n_faces):
            omega_hat = omega[i] - k[i] * u[j] - l[i] * v[j]
            if omega_hat <= abs(config.f):
                break

            if omega_hat >= N[j]:
                for p in range(j):
                    out[i, p] -= out[i, j - 1]

                break

            m = -np.sqrt(
                wvn_hor_sq * (N[j] ** 2 - omega_hat ** 2) /
                (omega_hat ** 2 - config.f ** 2)
            )

            cg_r = -m * (
                (omega_hat ** 2 - config.f ** 2) /
                omega_hat / (wvn_hor_sq + m ** 2)
            )

            threshold = rho[j] * omega_hat * (1 / m ** 2 + 1 / wvn_hor_sq) / 2
            out[i, j] = min(out[i, j - 1] / cg_r, threshold) * cg_r

    return out.T

@nb.njit
def interp(r: np.ndarray, z: np.ndarray, profile: np.ndarray) -> np.ndarray:
    """
    Interpolate data from the mean state to the ray volume positions. This
    function is still experimental, and may be abandoned if sufficient gains in
    performance over `np.interp` cannot be achieved.

    Parameters
    ----------
    r
        Ray volume vertical positions.
    z
        Vertical grid to interpolate from.
    profile
        Mean state data to interpolate.

    Returns
    -------
    np.ndarray
        Data from `profile` interpolated to ray volume vertical positions.

    """

    dz = z[1] - z[0]
    r = np.nan_to_num(np.clip(r, z[0], z[-1]))
    jdx = np.floor((r - z[0]) / dz).astype(np.int32)
    jdx = np.minimum(jdx, len(z) - 2)

    slopes = (profile[1:] - profile[:-1]) / dz
    out = profile[jdx] + slopes[jdx] * (r - z[jdx])
    
    return out

@nb.njit
def project(
    r: np.ndarray,
    dr: np.ndarray,
    edges: np.ndarray,
    data: np.ndarray
) -> np.ndarray:
    """
    Project data associated with each ray onto the vertical grid. Profiling
    finds that with numba, the fastest approach is to keep the logic for this
    function and `get_fracs` separate, and moreover to have the iteration logic
    repeated instead of reused.

    Parameters
    ----------
    r
        Position of ray volume centers.
    dr
        Ray volume extents.
    edges
        Edges of the vertical grid regions to project onto.
    data
        Data variables associated with each ray (e.g. momentum flux) to project.

    Returns
    -------
    np.ndarray
        Array of shape `(data.shape[0], len(edges) - 1)` of the projected values
        of at each grid point of each variable passed as a row of `data`.

    """

    r_lo = r - 0.5 * dr
    r_hi = r + 0.5 * dr

    shape = (data.shape[0], len(edges) - 1)
    proj = np.zeros(shape)

    for i, (a, b) in enumerate(zip(r_lo, r_hi)):
        if np.isnan(a):
            continue

        for j, (z_lo, z_hi) in enumerate(zip(edges[:-1], edges[1:])):
            if b < z_lo:
                break

            if z_hi < a:
                continue

            frac = (min(b, z_hi) - max(a, z_lo)) / (z_hi - z_lo)
            for k in range(data.shape[0]):
                proj[k, j] = proj[k, j] + frac * data[k, i]

    return proj
