import torch
import xarray as xr

from msgwam import config
from msgwam.constants import PROP_NAMES
from msgwam.dispersion import cg_r, m_from
from msgwam.utils import open_dataset

def get_flux(ds: xr.Dataset) -> xr.DataArray:
    """
    Compute the momentum flux of each ray volume in the dataset.

    Parameters
    ----------
    ds
        Dataset of ray volume properties.

    Returns
    -------
    xr.DataArray
        Momentum fluxes. Has the same coordinates as `ds`.

    """

    keep = PROP_NAMES[2:-2]
    k, l, m, dk, dl, dm, dens = torch.as_tensor(ds[keep].to_array().values)
    flux = k * (dens * dk * dl * dm) * cg_r(k, l, m)

    return xr.DataArray(flux, ds.coords)

def get_spectrum() -> xr.Dataset:
    """
    Return the ray volume properties of the spectrum as specified in the config.
    Functions in this file (excluding utilities) should return a `Dataset` with
    coordinates corresponding to time (optional) and phase speed and variables
    for each of the ray volume properties in `PROP_NAMES` (excluding r and dr,
    which are added by this function, and age and meta, which are added later
    during integration by the `RayCollection`).

    Returns
    -------
    xr.Dataset
        Dataset of ray volume properties.

    """

    ds: xr.Dataset = globals()['_' + config.spectrum_type]()

    if config.spectrum_type != 'custom':
        ds['dr'] = config.dr_init * xr.ones_like(ds['dens'])
        ds['r'] = config.r_ghost - 0.5 * ds['dr']
        ds = _coarsen(ds)

    return ds[PROP_NAMES[:-2]]

def _coarsen(ds: xr.Dataset) -> xr.Dataset:
    """
    Take a dataset of source spectrum data and regrid it to the phase speed grid
    specified by the config file, ensuring that total flux is conserved.

    Parameters
    ----------
    ds
        Dataset of ray volume properties, presumably with on a phase speed grid
        with more than `config.n_source` points.

    Returns
    -------
    xr.Dataset
        Coarsened dataset, unless the original dataset was already no finer than
        the config grid, in which case the original dataset is returned as is.
 
    """

    if len(ds['cp_x']) <= config.n_source:
        return ds

    targets = get_flux(ds)
    cs = _get_phase_speeds(config.n_source)
    kwargs = {'fill_value' : 'extrapolate'}
    ds = ds.interp(cp_x=cs, kwargs=kwargs)

    k = torch.as_tensor(ds['k'].values)
    l = torch.as_tensor(ds['l'].values)
    dc = cs[1] - cs[0]

    m = m_from(k, l, cs)
    dm = abs(k * dc / cg_r(k, l, m))
    ds['m'] = xr.DataArray(m, ds.coords)
    ds['dm'] = xr.DataArray(dm, ds.coords)

    bins = ds['cp_x'].sel(cp_x=targets['cp_x'], method='nearest')
    bins = bins.assign_coords(cp_x=targets['cp_x'])
    targets = targets.groupby(bins).sum()

    factors = targets / get_flux(ds)
    ds['dens'] = ds['dens'] * factors

    return ds

def _from_file() -> xr.Dataset:
    """Read and return a (possibly time-varying) source spectrum from disc."""

    return open_dataset(config.spectrum_file)

def _gaussians() -> xr.Dataset:
    """
    Constant-in-time source spectrum consisting of two Gaussian peaks, symmetric
    about the origin in phase space. If `config.c_center == 0`, the two peaks
    coincide with one another. Returned on a very fine phase speed grid so that
    the spectral data can be coarsened consistently prior to integration.
    """

    wvn_hor = 2 * torch.pi / config.wvl_hor_char
    phi = torch.deg2rad(torch.as_tensor(config.direction))
    k, l = wvn_hor * phi.cos(), wvn_hor * phi.sin()
    cs = _get_phase_speeds(int(1e5))

    dc = abs(cs[1] - cs[0])
    k = k * torch.sign(cs)
    ms = m_from(k, l, cs)

    dk, dl = config.dk_init, config.dl_init
    dms = abs(k * dc / cg_r(k, l, ms))

    fluxes = (-0.5 * ((abs(cs) - config.c_center) / config.c_width) ** 2).exp()
    fluxes = fluxes * config.bc_mom_flux / fluxes.sum()
    ones = torch.ones(len(cs))

    spectrum = torch.vstack((
        k, l * ones, ms,
        dk * ones, dl * ones, dms,
        fluxes / (k ** 2 * dk * dl * dc)
    ))

    data = {'cp_x' : cs}
    for i, prop in enumerate(PROP_NAMES[2:-2]):
        data[prop] = ('cp_x', spectrum[i])

    return xr.Dataset(data)

def _get_phase_speeds(n: int) -> torch.Tensor:
    """
    Compute a grid of phase speeds at ray volume centers.

    Parameters
    ----------
    n
        How many points should be in the grid. 

    Returns
    -------
    torch.Tensor
        Center of each source ray volume in phase speed.
 
    """

    bounds = torch.linspace(-config.c_max, config.c_max, n + 1)
    return (bounds[:-1] + bounds[1:]) / 2
