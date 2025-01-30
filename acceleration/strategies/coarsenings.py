import numpy as np
import xarray as xr

from msgwam import config
from msgwam.utils import get_vertical_grids

from ..hyperparameters import strategies as hp
from ..shared.distributed import product

from .integration import get_integration, get_overrides
from .utils import get_rmse, load_data

def get_coarse_errors(ref: xr.DataArray) -> xr.DataArray:
    """
    Get root-mean-square errors as a function of height for each coarsening.

    Parameters
    ----------
    ref
        Array of containing the reference flux time series.

    Returns
    -------
    xr.DataArray
        Array of errors with coordinates `'dr'` and `'n_source'` ranging over
        the grid of coarsenings, along with `'z_faces'` ranging over cell faces
        in the vertical grid.

    """

    drs, n_sources = _get_grid()
    profiles = np.zeros((len(drs), len(n_sources), config.n_grid))

    for i, dr in enumerate(drs):
        for j, n_source in enumerate(n_sources):
            flux = load_data(_get_path(dr, n_source))
            profiles[i, j] = get_rmse(ref, flux).values

    z, _ = get_vertical_grids()
    coords = {'dr' : drs, 'n_source' : n_sources, 'z_faces' : z}
    return xr.DataArray(profiles, coords)

def save_coarsenings() -> None:
    """
    Integrate over the grid of coarse resolutions and save each output dataset.
    """

    kwargs = get_overrides('coarse')
    for dr, n_source in product(*_get_grid()):
        kwargs['dr_init'] = float(dr)
        kwargs['n_source'] = n_source

        with config.override(**kwargs):
            ds = get_integration().mean('member')
            ds.to_netcdf(_get_path(dr, n_source))

def _get_grid() -> tuple[list[int], list[int]]:
    """
    Return lists of values for `config.dr_init` and `config.n_source` defining
    the grid over which to search for the optimal coarse configuration.

    Returns
    -------
    list[int], list[int]
        Values for `config.dr_init` and `config.n_source`, respectively.

    """

    drs = np.linspace(hp.dr_min, hp.dr_max, 3)
    n_sources = np.linspace(hp.n_source_min, hp.n_source_max, 3)
    drs, n_sources = drs.astype(int), n_sources.astype(int)[::-1]

    return drs.tolist(), n_sources.tolist()

def _get_path(dr: int, n_source: int) -> str:
    """
    Get the path where each coarse-resolution integration should be saved.
    """

    data_dir = f'data/{config.name}/coarsenings'
    return f'{data_dir}/dr-{dr}_n-source-{n_source}.nc'
