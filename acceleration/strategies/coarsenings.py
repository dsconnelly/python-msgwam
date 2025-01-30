import numpy as np

from msgwam import config

from ..hyperparameters import strategies as hp
from ..shared.distributed import product

from .integration import get_integration, get_overrides

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

    drs = np.linspace(hp.dr_min, hp.dr_max, 10)
    n_sources = np.linspace(hp.n_source_min, hp.n_source_max, 10)
    drs, n_sources = drs.astype(int), n_sources.astype(int)[::-1]

    return drs.tolist(), n_sources.tolist()

def _get_path(dr: int, n_source: int) -> str:
    """
    Get the path where each coarse-resolution integration should be saved.
    """

    data_dir = f'data/{config.name}/coarsenings'
    return f'{data_dir}/dr-{dr}_n-source-{n_source}.nc'
