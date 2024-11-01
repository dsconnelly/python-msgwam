import numba as nb
import numpy as np

@nb.njit
def get_fracs(
    r: np.ndarray,
    dr: np.ndarray,
    edges: np.ndarray
) -> np.ndarray:
    """
    Compute the fraction of each grid cell intersected by each ray.

    Parameters
    ----------
    r
        Position of ray volume centers.
    dr
        Ray volume extents.
    edges
        Edges of the vertical grid regions to project onto.

    Returns
    -------
    np.ndarray
        Array of fractions with shape `(len(edges) - 1, len(r))` such that the
        value at [i, j] is the fraction of cell i intersected by ray j.

    """

    r_lo = r - 0.5 * dr
    r_hi = r + 0.5 * dr

    fracs = np.zeros((len(r), len(edges) - 1))
    for i, (a, b) in enumerate(zip(r_lo, r_hi)):
        if np.isnan(a):
            continue

        for j, (z_lo, z_hi) in enumerate(zip(edges[:-1], edges[1:])):
            if b < z_lo:
                break

            if z_hi < a:
                continue

            fracs[i, j] = (min(b, z_hi) - max(a, z_lo)) / (z_hi - z_lo)

    return fracs.T

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
        Data associated with each ray (e.g. momentum flux) to project.

    Returns
    -------
    np.ndarray
        Array with `len(edges) - 1` elements containing the projected profile.

    """

    r_lo = r - 0.5 * dr
    r_hi = r + 0.5 * dr

    proj = np.zeros((len(edges) - 1))
    for i, (a, b) in enumerate(zip(r_lo, r_hi)):
        if np.isnan(a):
            continue

        for j, (z_lo, z_hi) in enumerate(zip(edges[:-1], edges[1:])):
            if b < z_lo:
                break

            if z_hi < a:
                continue

            frac = (min(b, z_hi) - max(a, z_lo)) / (z_hi - z_lo)
            proj[j] = proj[j] + frac * data[i]

    return proj
