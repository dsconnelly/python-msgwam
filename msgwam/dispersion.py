import torch

from . import config

def cg_r(
    k: torch.Tensor,
    l: torch.Tensor,
    m: torch.Tensor
) -> torch.Tensor:
    """
    Calculate the vertical group velocity of internal gravity waves.

    Parameters
    ----------
    k
        Tensor of zonal wavenumbers.
    l
        Tensor of meridional wavenumbers.
    m
        Tensor of vertical wavenumbers.

    Returns
    -------
    torch.Tensor
        Tensor of vertical group velocities.
        
    """

    wvn_sq = k ** 2 + l ** 2 + m ** 2
    _omega_hat = omega_hat(k, l, m)

    return -m * (
        (_omega_hat ** 2 - config.f0 ** 2) /
        (_omega_hat * wvn_sq)
    )

def cp_x(
    k: torch.Tensor,
    l: torch.Tensor,
    m: torch.Tensor
) -> torch.Tensor:
    """
    Calculate the zonal phase velocity of internal gravity waves.

    Parameters
    ----------
    k
        Tensor of zonal wavenumbers.
    l
        Tensor of meridional wavenumbers.
    m
        Tensor of vertical wavenumbers.

    Returns
    -------
    torch.Tensor
        Tensor of zonal phase velocities.
        
    """

    return omega_hat(k, l, m) / k

def m_from(
    k: torch.Tensor,
    l: torch.Tensor,
    cp_x: torch.Tensor
) -> torch.Tensor:
    """
    Calculate the vertical wavenumber of internal gravity waves if the zonal
    phase velocity is known.

    Parameters
    ----------
    k
        Tensor of zonal wavenumbers.
    l
        Tensor of meridional wavenumbers.
    cp_x
        Tensor of zonal phase velocities.

    Returns
    -------
    torch.Tensor
        Tensor of intrinsic frequencies.
        
    """

    return -torch.sqrt(
        (k ** 2 + l ** 2) * (config.N0 ** 2 - cp_x ** 2 * k ** 2) /
        (cp_x ** 2 * k ** 2 - config.f0 ** 2)
    )

def omega_hat(
    k: torch.Tensor,
    l: torch.Tensor,
    m: torch.Tensor
) -> torch.Tensor:
    """
    Calculate the intrinsic frequency of internal gravity waves.

    Parameters
    ----------
    k
        Tensor of zonal wavenumbers.
    l
        Tensor of meridional wavenumbers.
    m
        Tensor of vertical wavenumbers.

    Returns
    -------
    torch.Tensor
        Tensor of intrinsic frequencies.
        
    """

    return torch.sqrt(
        (config.N0 ** 2 * (k ** 2 + l ** 2) + config.f0 ** 2 * m ** 2) /
        (k ** 2 + l ** 2 + m ** 2)
    )
