import torch

from . import config
from .dispersion import cg_r, m_from

def get_spectrum() -> torch.Tensor:
    """
    Return the ray volume properties of the spectrum specified in the config.

    Returns
    -------
    torch.Tensor
        Tensor whose rows correspond to ray volume properties and whose columns
        correspond to individual ray volumes in the spectrum.

    """

    return globals()['_' + config.spectrum_type]()

def _gaussians() -> torch.Tensor:
    """Spectrum consisting of Gaussian peaks symmetric about the origin."""

    dr = config.dr_init
    r = config.r_ghost - 0.5 * dr

    wvn_hor = 2 * torch.pi / config.wvl_hor_char
    direction = torch.deg2rad(torch.tensor(config.direction))
    k = wvn_hor * torch.cos(direction)
    l = wvn_hor * torch.sin(direction)

    n_c = config.n_source // 2
    c_max = config.c_center + 2 * config.c_width
    cp_xs = (torch.arange(n_c) + 0.5) * c_max / (n_c - 0.5)
    ms = m_from(k, l, cp_xs)
    
    dk = config.dk_init
    dl = config.dl_init
    dc = cp_xs[1] - cp_xs[0]
    dms = abs(k * dc / cg_r(k, l, ms))

    fluxes = torch.exp(-((cp_xs - config.c_center) / config.c_width) ** 2)
    fluxes = (config.bc_mom_flux / fluxes.sum()) * fluxes / 2
    ones = torch.ones(config.n_source // 2)
    
    data = torch.vstack((
        r * ones, dr * ones,
        k * ones, l * ones, ms,
        dk * ones, dl * ones, dms,
        fluxes / (k ** 2 * dk * dl * dc)
    ))

    data = torch.hstack((data, data))
    data[2, (config.n_source // 2):] *= -1

    return data
