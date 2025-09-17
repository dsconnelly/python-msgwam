import numpy as np
import xarray as xr

from msgwam import config
from msgwam.utils import gaussian_filter, get_vertical_grids

from .. import hyperparameters as hp
from ..shared.constants import ACCEL_HOURS, RMS_FILTERS, STRAT_FILTERS
from ..shared.distributed import product

from .integration import get_integration, get_overrides
from .utils import by_kind, get_rmse, get_rnames, load_data

_Z_CUTOFF = 20e3

def get_global_scores(kind: str, *rnames: str) -> xr.DataArray:
    """
    Get the normalized errors averaged across multiple runs for each component.

    Parameters
    ----------
    kind
        What field to compute the scores for.
    rnames
        Names of runs to average over. If not provided, only the data from the
        currently-loaded configuration will be used.

    Returns
    -------
    xr.DataArray
        Array with coordinates `('component', 'dr', 'n_source')` giving the
        run- and height-averaged normalized errors for each resolution.

    """

    if not rnames:
        rnames = [config.name]

    error = 0
    for rname in rnames:
        fname = f'coarse-errors-{kind}.nc'
        ds = xr.open_dataset(f'data/{rname}/coarsenings/{fname}')
        add = np.minimum(1, ds['error'] / ds['rms'])
        error += add.mean('z', skipna=True)

    return error / len(rnames)

def save_coarsenings() -> None:
    """
    Integrate each pair in the grid of candidate coarse resolutions and save
    each output dataset to disk. Also save a dataset containing the RMS errors
    of each configuration.

    """

    kwargs = get_overrides('coarse')
    for dr, n_source in product(*_get_grid()):
        kwargs['dr_source'] = float(dr)
        kwargs['n_source'] = n_source

        with config.override(**kwargs):
            ds = get_integration().mean('member')
            ds.to_netcdf(_get_path(dr, n_source))

def update_config(kind: str, prefix: str='') -> None:
    """
    Update the values of `dr_source` and `n_source` in one or more configuration
    files to the best values found during the grid search. Can take into account
    grid searches across multiple runs.

    Parameters
    ----------
    kind
        What field to use to select the best coarsening.
    prefix
        Prefix to match to find runs to update.

    """

    rnames = get_rnames(prefix)
    scores = get_global_scores(kind, *rnames).mean('component')
    i, j = (da.item() for da in scores.argmin(...).values())
    drs, n_sources = _get_grid()

    for rname in rnames:
        with open(f'config/{rname}.toml') as f:
            lines = f.readlines()

        with open(f'config/{rname}.toml', 'w') as f:
            for line in lines:
                if line.startswith('dr_source'):
                    f.write(f'dr_source = {drs[i]}\n')

                elif line.startswith('n_source'):
                    f.write(f'n_source = {n_sources[j]}\n')

                else:
                    f.write(line)

@by_kind
def save_coarse_errors(kind: str) -> None:
    """
    Get the root-mean-square errors as a function of height for each coarsening.

    Parameters
    ----------
    kind
        What field to compute the errors in.

    """

    drs, n_sources = _get_grid()
    components = list(hp.scenarios.components)
    z = get_vertical_grids()[kind != 'flux']

    rms = np.zeros((len(components), len(z)))
    error = np.zeros((len(components), len(drs), len(n_sources), len(z)))

    for k, c in enumerate(components):
        field = 'uv'[k] if kind == 'wind' else f'{kind}_{c}'

        ref = load_data(
            'reference',
            field=field,
            time_filter=None,
            z_filter=None
        )

        drop = z < _Z_CUTOFF
        drop[-config.n_sponge:] = True

        filters = STRAT_FILTERS.copy()
        if kind == 'acceleration':
            filters['hours'] = ACCEL_HOURS

        tmp = gaussian_filter(ref, **RMS_FILTERS)
        ref = gaussian_filter(ref, **filters)
        rms[k] = get_rmse(tmp).values

        for i, dr in enumerate(drs):
            for j, n_source in enumerate(n_sources):
                kwargs = dict(time_filter=(filters['hours'] * 3600))
                flux = load_data(_get_path(dr, n_source), field, **kwargs)
                error[k, i, j] = get_rmse(ref, flux).values
                error[k, i, j, drop] = np.nan

    xr.Dataset({
        'component' : components,
        'dr' : drs, 'n_source' : n_sources, 'z' : z,
        'error' : (('component', 'dr', 'n_source', 'z'), error),
        'rms' : (('component', 'z'), rms)
    }).to_netcdf(f'data/{config.name}/coarsenings/coarse-errors-{kind}.nc')

def _get_grid() -> tuple[list[int], list[int]]:
    """
    Return lists of values for `config.dr_source` and `config.n_source` within
    which to search for the optimal coarse configuration.

    Returns
    -------
    list[int], list[int]
        Values for `config.dr_source` and `config.n_source`, respectively.

    """

    _hp = hp.strategies
    drs = np.linspace(_hp.dr_min, _hp.dr_max, _hp.n_dr)
    n_sources = np.linspace(_hp.n_source_min, _hp.n_source_max, _hp.n_n_source)
    drs, n_sources = drs.astype(int), n_sources.astype(int)[::-1]

    return drs.tolist(), n_sources.tolist()

def _get_path(dr: int, n_source: int) -> str:
    """
    Get the path where each coarse-resolution integration should be saved.

    Parameters
    ----------
    dr, n_source
        Integers describing the current coarse integration.

    Returns
    -------
    str
        Path for the netCDF output.

    """

    data_dir = f'data/{config.name}/coarsenings'
    return f'{data_dir}/dr-{dr}_n-source-{n_source}.nc'
