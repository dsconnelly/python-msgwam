from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from . import config
from .rays import _cg_r, _omega_hat

if TYPE_CHECKING:
    from .mean import MeanFlow

# Each function in this module should accept a MeanFlow and return an array with
# rows corresponding to the ray properties named in RayCollection.props and
# columns corresponding the ray volumes that should be launched at the source.

def desaubies(mean: MeanFlow) -> np.ndarray:
    c_grid = np.linspace(*config.c_bounds, 2 * config.n_c + 1)
    omega_grid = np.linspace(*config.omega_bounds, 2 * config.n_omega + 1)
    c_grid, omega_grid = c_grid[1:-1:2], omega_grid[1:-1:2]
    c, omega = np.meshgrid(c_grid, omega_grid)
    c, omega = c.flatten(), omega.flatten()

    dc = c_grid[1] - c_grid[0]
    domega = omega_grid[1] - omega_grid[0]
    dphi = np.pi / 2

    m = -config.N0 / c
    wvn_hor = omega / c
    dm = dc * m ** 2 / config.N0
    m_star = -2 * np.pi / config.wvl_ver_char

    top = c * config.N0 ** 3 * omega ** (-2 / 3)
    bottom = config.N0 ** 4 + m_star ** 4 * c ** 4
    F0 = m_star ** 3 * top / bottom

    n_each = config.n_c * config.n_omega
    data = np.zeros((9, 4 * n_each))

    dr = np.ones(n_each) * config.dr_init
    r = np.ones(n_each) * (config.r_ghost - 0.5 * config.dr_init)
    rhobar = np.interp(r[0], mean.r_centers, mean.rho)
    
    C = config.bc_mom_flux / (4 * rhobar * F0.sum() * dc * domega * dphi)
    F3 = C * config.N0 ** 3 * F0 / (m ** 4 * omega)

    for i in range(4):
        phi = i * dphi
        k = wvn_hor * np.round(np.cos(phi))
        l = wvn_hor * np.round(np.sin(phi))

        dk = domega * abs(m) / config.N0
        dl = wvn_hor * dphi

        if i % 2 == 1:
            dk, dl = dl, dk

        cg_r = _cg_r(k, l, m)
        dens = rhobar * F3 / (wvn_hor * cg_r)
        chunk = np.vstack((r, dr, k, l, m, dk, dl, dm, dens))
        data[:, (i * n_each):((i + 1) * n_each)] = chunk

    return data

def legacy(mean: MeanFlow) -> np.ndarray:
    """
    Calculate source ray volumes as was done in the original version of this
    Python code. Note that the library defaults to not launching more rays at
    the lower boundary when this option is chosen.
    """
    
    wvn_hor = 2 * np.pi / config.wvl_hor_char
    direction = np.deg2rad(config.direction)

    k = wvn_hor * np.cos(direction) * np.ones(config.n_ray_max)
    l = wvn_hor * np.sin(direction) * np.ones(config.n_ray_max)
    m = -2 * np.pi / config.wvl_ver_char * np.ones(config.n_ray_max)

    r_min, r_max = config.r_init_bounds
    r_edges = np.linspace(r_min, r_max, config.n_ray_max + 1)
    dr = r_edges[1] - r_edges[0] * np.ones(config.n_ray_max)
    r = (r_edges[:-1] + r_edges[1:]) / 2

    dk = config.dk_init * np.ones(config.n_ray_max)
    dl = config.dl_init * np.ones(config.n_ray_max)
    dm = config.r_m_area / dr

    rhobar = np.interp(r, mean.r_centers, mean.rho)
    omega_hat = _omega_hat(k=k, l=l, m=m)

    amplitude = (
        (config.alpha ** 2 * rhobar * omega_hat * config.N0 ** 2) /
        (2 * m ** 2 * (omega_hat ** 2 - config.f0 ** 2))
    )

    profile = np.exp(-0.5 * ((r - r.mean()) / 2000) ** 2)
    dens = amplitude * profile / (dk * dl * dm)

    return np.vstack((r, dr, k, l, m, dk, dl, dm, dens))

def gaussians(_) -> np.ndarray:
    dr = config.dr_init
    r = config.r_ghost - 0.5 * dr

    wvn_hor = 2 * np.pi / config.wvl_hor_char
    direction = np.deg2rad(config.direction)
    k = wvn_hor * np.cos(direction)
    l = wvn_hor * np.sin(direction)

    c_max = config.c_center + 2 * config.c_width
    c_min = max(config.c_center - 2 * config.c_width, 0.5)
    c_bounds = np.linspace(c_min, c_max, (config.n_source // 2) + 1)
    m_bounds = _m_from(k, l, c_bounds)

    ms = (m_bounds[:-1] + m_bounds[1:]) / 2
    dms = abs(m_bounds[1:] - m_bounds[:-1])
    cs = _c_from(k, l, ms)

    fluxes = np.exp(-(((cs - config.c_center) / config.c_width) ** 2))
    fluxes = (config.bc_mom_flux / fluxes.sum()) * fluxes / 2

    dk = config.dk_init
    dl = config.dl_init

    data = np.zeros((9, config.n_source))
    for j, (m, dm, flux) in enumerate(zip(ms, dms, fluxes)):
        volume = abs(dk * dl * dm)
        cg_r = _cg_r(k=k, l=l, m=m)
        dens = flux / abs(k * volume * cg_r)

        data[:, j] = np.array([r, dr, k, l, m, dk, dl, dm, dens])
        data[:, j + (config.n_source // 2)] = data[:, j]
        data[2, j + (config.n_source // 2)] *= -1

    return data

def _c_from(k: np.ndarray, l: np.ndarray, m: np.ndarray) -> np.ndarray:
    top = config.N0 ** 2 * (k ** 2 + l ** 2) + config.f0 ** 2 * m ** 2
    bottom = (k ** 2 + l ** 2 + m ** 2) * k ** 2

    return np.sqrt(top / bottom)

def _m_from(k: np.ndarray, l: np.ndarray, c: np.ndarray) -> np.ndarray:
    top = (k ** 2 + l ** 2) * (config.N0 ** 2 - c ** 2 * k ** 2)
    bottom = (c ** 2 * k ** 2 - config.f0 ** 2)

    return -np.sqrt(top / bottom)
