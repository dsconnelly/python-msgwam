import numpy as np
import xarray as xr

from msgwam import config
from msgwam.means import PrescribedWind
from msgwam.propagators import TransientPropagator
from msgwam.utils import open_dataset

def get_min_dr(round_to: int=25) -> float:
    """
    Calculate the minimum dr value that can be resolved by the bottom boundary
    condition, given some configuration overrides.

    Parameters
    ----------
    round_to
        What to round the returned value up to.

    Returns
    -------
    float
        Minimum resolvable dr.

    """

    mean = PrescribedWind()
    cg = TransientPropagator(mean)._get_cg_r(mean)
    distance = np.nanmax(cg * config.dt)

    return np.ceil(distance / round_to) * round_to

def get_rmse(a: xr.DataArray, b: xr.DataArray | float = 0) -> xr.DataArray:
    """
    Compute the root-mean-square error over time between two arrays. The second
    argument can also be passed in as a constant float, so that this function
    can be used to calculate the RMS value of the data by passing in zero.

    Parameters
    ----------
    a, b
        Data with which to compute RMS errors.

    Returns
    -------
    xr.DataArray
        Array of RMS errors, with the time dimension averaged out.

    """

    return np.sqrt(((a - b) ** 2).mean('time'))

def load_flux(path: str, z_faces: np.ndarray) -> xr.DataArray:
    """
    Load the zonal gravity wave momentum flux from a netCDF file. This function
    adds the easterly and westerly components and ensures that the returned flux
    is at the correct temporal and vertical resolution.

    Parameters
    ----------
    path
        Path to netCDF file containing flux data.
    z_faces
        Array of vertical grid faces to interpolate onto.

    Returns
    -------
    xr.DataArray
        Postprocessed flux time series.

    """

    with open_dataset(path) as ds:
        ds = ds.interp(z_faces=z_faces)
        ds = ds.resample(time='3h').mean('time')
        flux = ds['pmf_e'] + ds['pmf_w']

    return flux