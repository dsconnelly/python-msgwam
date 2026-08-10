from typing import Literal

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

# `base` selects which `n_max` regime the grid search is calibrating: `'coarse'`
# is the original n_max = 250 search (baked into `config/*.toml`), `'mima'` is
# the n_max = 2500 ("MiMAlike") search (baked into `hyperparameters/*.toml`,
# since it is not the default source resolution and so cannot share the
# `dr_source`/`n_source` fields already used by the `coarse` strategy).
_BASE_STRATEGIES = {'coarse' : 'coarse', 'mima' : 'MiMAlike'}

def get_global_scores(
    kind: str,
    *rnames: str,
    base: Literal['coarse', 'mima']='coarse'
) -> xr.DataArray:
    """
    Get the normalized errors averaged across multiple runs for each component.

    Parameters
    ----------
    kind
        What field to compute the scores for.
    rnames
        Names of runs to average over. If not provided, only the data from the
        currently-loaded configuration will be used.
    base
        Which grid search to score: `'coarse'` (n_max = 250) or `'mima'`
        (n_max = 2500).

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
        fname = f'grid-search-errors-{kind}.nc'
        ds = xr.open_dataset(f'{get_grid_search_dir(rname, base)}/{fname}')
        add = np.minimum(1, ds['error'] / ds['rms'])
        error += add.mean('z', skipna=True)

    return error / len(rnames)

def save_grid_search(base: Literal['coarse', 'mima']='coarse') -> None:
    """
    Integrate each pair in the grid of candidate source resolutions and save
    each output dataset to disk. Also save a dataset containing the RMS errors
    of each configuration.

    Parameters
    ----------
    base
        Which regime to search over: `'coarse'` (n_max = 250, overrides taken
        from the `coarse` strategy) or `'mima'` (n_max = 2500, overrides taken
        from the `MiMAlike` strategy). Each base has its own grid of
        `dr_source`/`n_source` candidates from `_get_grid`.

    """

    kwargs = get_overrides(_BASE_STRATEGIES[base])
    for dr, n_source in product(*_get_grid(base)):
        kwargs['dr_source'] = float(dr)
        kwargs['n_source'] = n_source

        with config.override(**kwargs):
            ds = get_integration().mean('member')
            ds.to_netcdf(_get_path(dr, n_source, base))

def update_config(
    kind: str,
    prefix: str='',
    base: Literal['coarse', 'mima']='coarse'
) -> None:
    """
    Update the values of `dr_source` and `n_source` to the best values found
    during the grid search. Can take into account grid searches across
    multiple runs.

    Parameters
    ----------
    kind
        What field to use to select the best source resolution.
    prefix
        Prefix to match to find runs to update.
    base
        Which grid search to read: `'coarse'` (n_max = 250) or `'mima'`
        (n_max = 2500). For `'coarse'`, the best `dr_source`/`n_source` are
        written directly into `config/<rname>.toml`, since those fields are
        read as-is by the `coarse` strategy. For `'mima'`, they are written
        into the `[strategies]` section of `hyperparameters/<rname>.toml` as
        `mima_dr_source`/`mima_n_source`, which the `MiMAlike` strategy reads
        in `'calibrated'` mode, so as not to disturb the `coarse` calibration
        already baked into the same `config/<rname>.toml`.

    """

    rnames = get_rnames(prefix)
    scores = get_global_scores(kind, *rnames, base=base).sel(component='x')
    i, j = (da.item() for da in scores.argmin(...).values())
    drs, n_sources = _get_grid(base)

    if base == 'coarse':
        for rname in rnames:
            _update_config_file(rname, drs[i], n_sources[j])

    else:
        for rname in rnames:
            _update_hyperparameters_file(rname, drs[i], n_sources[j])

@by_kind
def save_grid_search_errors(
    kind: str,
    base: Literal['coarse', 'mima']='coarse'
) -> None:
    """
    Get the root-mean-square errors as a function of height for each candidate
    source resolution in the grid search.

    Parameters
    ----------
    kind
        What field to compute the errors in.
    base
        Which grid search to score: `'coarse'` (n_max = 250) or `'mima'`
        (n_max = 2500).

    """

    drs, n_sources = _get_grid(base)
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
                path = _get_path(dr, n_source, base)
                flux = load_data(path, field, **kwargs)
                error[k, i, j] = get_rmse(ref, flux).values
                error[k, i, j, drop] = np.nan

    xr.Dataset({
        'component' : components,
        'dr' : drs, 'n_source' : n_sources, 'z' : z,
        'error' : (('component', 'dr', 'n_source', 'z'), error),
        'rms' : (('component', 'z'), rms)
    }).to_netcdf(f'{get_grid_search_dir(config.name, base)}/grid-search-errors-{kind}.nc')

def _get_grid(base: Literal['coarse', 'mima']='coarse') -> tuple[list[int], list[int]]:
    """
    Return lists of values for `config.dr_source` and `config.n_source` within
    which to search for the optimal source resolution.

    Parameters
    ----------
    base
        Whether the grid search is being done for the coarse (n_max = 250) or
        MiMAlike (n_max = 2500) setup.

    Returns
    -------
    list[int], list[int]
        Values for `config.dr_source` and `config.n_source`, respectively.

    """

    _hp = hp.strategies

    if base == 'coarse':
        dr_min, dr_max = _hp.dr_min, _hp.dr_max
        n_source_min, n_source_max = _hp.n_source_min, _hp.n_source_max

    else:
        dr_min, dr_max = 100, 2100
        n_source_min, n_source_max = 24, 240

    drs = np.linspace(dr_min, dr_max, _hp.n_dr)
    n_sources = np.linspace(n_source_min, n_source_max, _hp.n_n_source)
    drs, n_sources = drs.astype(int), n_sources.astype(int)[::-1]

    return drs.tolist(), n_sources.tolist()

def get_grid_search_dir(rname: str, base: Literal['coarse', 'mima']) -> str:
    """
    Get the directory where grid search output for a given run and base
    (n_max regime) is stored. Both bases get an explicit suffix, so that
    neither directory name implies it is somehow the default.

    Parameters
    ----------
    rname
        Name of the run (i.e. `config.name` when the search was run).
    base
        Which regime's directory to return.

    Returns
    -------
    str
        Path to the relevant directory.

    """

    return f'data/{rname}/grid-search-{base}'

def _get_path(dr: int, n_source: int, base: Literal['coarse', 'mima']) -> str:
    """
    Get the path where each grid search integration should be saved.

    Parameters
    ----------
    dr, n_source
        Integers describing the current candidate source resolution.
    base
        Which regime's grid search this integration belongs to.

    Returns
    -------
    str
        Path for the netCDF output.

    """

    return f'{get_grid_search_dir(config.name, base)}/dr-{dr}_n-source-{n_source}.nc'

def _update_config_file(rname: str, dr: int, n_source: int) -> None:
    """
    Rewrite the `dr_source` and `n_source` lines of `config/<rname>.toml` in
    place with the best values found during the `'coarse'` grid search.

    Parameters
    ----------
    rname
        Name of the configuration file (without extension) to update.
    dr, n_source
        Best values found during the grid search.

    """

    path = f'config/{rname}.toml'
    with open(path) as f:
        lines = f.readlines()

    with open(path, 'w') as f:
        for line in lines:
            if line.startswith('dr_source'):
                f.write(f'dr_source = {dr}\n')

            elif line.startswith('n_source'):
                f.write(f'n_source = {n_source}\n')

            else:
                f.write(line)

def _update_hyperparameters_file(rname: str, dr: int, n_source: int) -> None:
    """
    Rewrite the `mima_dr_source` and `mima_n_source` settings in the
    `[strategies]` section of `hyperparameters/<rname>.toml` in place with the
    best values found during the `'mima'` grid search. Unlike `dr_source` and
    `n_source` in `config/<rname>.toml`, these keys are not guaranteed to
    already be present in the file, so they are inserted at the end of the
    `[strategies]` section the first time this is called.

    Parameters
    ----------
    rname
        Name of the hyperparameter file (without extension) to update.
    dr, n_source
        Best values found during the grid search.

    """

    updates = {'mima_dr_source' : dr, 'mima_n_source' : n_source}
    path = f'hyperparameters/{rname}.toml'

    with open(path) as f:
        lines = f.readlines()

    def _flush_missing(output: list[str], found: set[str]) -> None:
        for key in updates.keys() - found:
            output.append(f'{key} = {updates[key]}\n')
            found.add(key)

    output: list[str] = []
    found: set[str] = set()
    in_strategies = False

    for line in lines:
        stripped = line.strip()

        if stripped.startswith('['):
            if in_strategies:
                _flush_missing(output, found)

            in_strategies = stripped == '[strategies]'
            output.append(line)
            continue

        key = stripped.split('=')[0].strip()
        if in_strategies and key in updates:
            output.append(f'{key} = {updates[key]}\n')
            found.add(key)

        else:
            output.append(line)

    if in_strategies:
        _flush_missing(output, found)

    with open(path, 'w') as f:
        f.writelines(output)
