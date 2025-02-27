import numpy as np
import xarray as xr

from msgwam import config
from msgwam.utils import get_vertical_grids

from .. import hyperparameters as hp
from ..shared.distributed import product

from .integration import get_integration, get_overrides
from .utils import get_rmse, load_data

def get_coarse_errors() -> xr.Dataset:
    """
    Get the root-mean-square errors as a function of height for each coarsening.
    If both zonal and meridional fluxes are considered, then the returned data
    will be the average error over both components.

    Returns
    -------
    xr.Dataset
        Dataset with coordinates `'dr'` and `'n_source'` ranging over the grid
        of coarsenings, along with `'z_faces'` ranging over cell faces in the
        vertical grid. Also includes the RMS flux itself at each level.

    """

    z, _ = get_vertical_grids()
    drs, n_sources = _get_grid()
    components = list(hp.scenarios.components)

    rms = np.zeros((len(components), len(z)))
    error = np.zeros((len(components), len(drs), len(n_sources), len(z)))

    for k, c in enumerate(components):
        ref = load_data('reference', f'flux_{c}')
        rms[k] = get_rmse(ref).values

        for i, dr in enumerate(drs):
            for j, n_source in enumerate(n_sources):
                flux = load_data(_get_path(dr, n_source), f'flux_{c}')
                error[k, i, j] = get_rmse(ref, flux).values

    return xr.Dataset({
        'component' : components,
        'dr' : drs, 'n_source' : n_sources, 'z_faces' : z,
        'error' : (('component', 'dr', 'n_source', 'z_faces'), error),
        'rms' : (('component', 'z_faces'), rms)
    })

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

def update_config() -> None:
    """
    Update the values of `dr_init` and `n_source` in the loaded configuration
    file to the best values found during the grid search.
    """

    ds = get_coarse_errors()
    errors = (ds['error'] / ds['rms']).mean(['component', 'z_faces'])
    i, j = (da.item() for da in errors.argmin(...).values())

    with open(f'config/{config.name}.toml') as f:
        lines = f.readlines()

    drs, n_sources = _get_grid()
    with open(f'config/{config.name}.toml', 'w') as f:
        for line in lines:
            if line.startswith('dr_init'):
                f.write(f'dr_init = {drs[i]}\n')

            elif line.startswith('n_source'):
                f.write(f'n_source = {n_sources[j]}\n')

            else:
                f.write(line)

def _get_grid() -> tuple[list[int], list[int]]:
    """
    Return lists of values for `config.dr_init` and `config.n_source` defining
    the grid over which to search for the optimal coarse configuration.

    Returns
    -------
    list[int], list[int]
        Values for `config.dr_init` and `config.n_source`, respectively.

    """

    _hp = hp.strategies
    drs = np.linspace(_hp.dr_min, _hp.dr_max, _hp.n_dr)
    n_sources = np.linspace(_hp.n_source_min, _hp.n_source_max, _hp.n_n_source)
    drs, n_sources = drs.astype(int), n_sources.astype(int)[::-1]

    return drs.tolist(), n_sources.tolist()

def _get_path(dr: int, n_source: int) -> str:
    """
    Get the path where each coarse-resolution integration should be saved.
    """

    data_dir = f'data/{config.name}/coarsenings'
    return f'{data_dir}/dr-{dr}_n-source-{n_source}.nc'
