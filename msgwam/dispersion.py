import numpy as np
import torch

from . import config

def get_cg_r(
    k: np.ndarray,
    l: np.ndarray,
    m: np.ndarray,
    N: float | np.ndarray
) -> np.ndarray:
    """
    Calculate the vertical group velocities of internal gravity waves.

    Parameters
    ----------
    k, l, m
        Arrays of zonal, meridional, and vertical wavenumbers, respectively.
    N
        Buoyancy frequency or array of buoyancy frequencies.

    Returns
    -------
    np.ndarray
        Array of vertical group velocities.

    """

    wvn_sq = k ** 2 + l ** 2 + m ** 2 + get_gamma() ** 2
    omega_hat = get_omega_hat(k, l, m, N)

    return -m * (omega_hat ** 2 - config.f ** 2) / omega_hat / wvn_sq

def get_cp_x(
    k: np.ndarray,
    l: np.ndarray,
    m: np.ndarray,
    N: float | np.ndarray
) -> np.ndarray:
    """
    Calculate the zonal phase velocities of internal gravity waves.

    Parameters
    ----------
    k, l, m
        Arrays of zonal, meridional, and vertical wavenumbers, respectively.
    N
        Buoyancy frequency or array of buoyancy frequencies.

    Returns
    -------
    np.ndarray
        Array of zonal phase velocities.

    """

    return get_omega_hat(k, l, m, N) / k

def get_dm(
    m: np.ndarray,
    dc: float | np.ndarray,
    N: float | np.ndarray
) -> np.ndarray:
    """
    Get the vertical wavenumber extent given the zonal phase velocities and the
    vertical wavenumbers themselves. Makes the hydrostatic approximation.

    Parameters
    ----------
    m
        Array of vertical wavenumbers.
    dc
        Extent or array of extents in phase velocity.
    N
        Buoyancy frequency or array of buoyancy frequencies.

    Returns
    -------
    np.ndarray
        Array of vertical wavenumber extents.

    """

    return dc * m ** 2 / N

def get_gamma() -> float:
    """
    Compute the scale height correction term.

    Returns
    -------
    float
        Scale height correction.

    """

    return (1 / 2 - 2 / 7) / config.H_rho

def get_m(
    k: np.ndarray,
    l: np.ndarray,
    omega_hat: np.ndarray,
    N: float | np.ndarray
) -> np.ndarray:
    """
    Calculate the vertical wavenumber of internal gravity waves, assuming the
    horizontal wavenumbers and the intrinsic frequency are known.

    Parameters
    ----------
    k, l
        Arrays of zonal and meridional wavenumbers, respectively.
    omega_hat
        Array of intrinsic frequencies.
    N
        Buoyancy frequency or array of buoyancy frequencies.

    Returns
    -------
    np.ndarray
        Array of vertical wavenumbers
    
    """

    omega_hat_sq = omega_hat ** 2

    return -_sqrt(
        (k ** 2 + l ** 2) * (N ** 2 - omega_hat_sq) /
        (omega_hat_sq - config.f ** 2)
    )

def get_omega_hat(
    k: np.ndarray,
    l: np.ndarray,
    m: np.ndarray,
    N: float | np.ndarray
) -> np.ndarray:
    """
    Calculate the intrinsic frequency of internal gravity waves.

    Parameters
    ----------
    k, l, m
        Arrays of zonal, meridional, and vertical wavenumbers, respectively.
    N
        Buoyancy frequency or array of buoyancy frequencies.

    Returns
    -------
    np.ndarray
        Array of intrinsic frequencies.

    """

    m2 = m ** 2 + get_gamma() ** 2

    return _sqrt(
        (N ** 2 * (k ** 2 + l ** 2) + config.f ** 2 * m2) /
        (k ** 2 + l ** 2 + m2)
    )

def _sqrt(a: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """
    Square root function that uses `torch` only when appropriate.

    Parameters
    ----------
    a
        Array or tensor to take the square root of.

    Returns
    -------
    np.ndarray | torch.Tensor
        Array or tensor of square roots, with type matching that of `a`.

    """

    if isinstance(a, torch.Tensor):
        return torch.sqrt(a)
    
    return np.sqrt(a)