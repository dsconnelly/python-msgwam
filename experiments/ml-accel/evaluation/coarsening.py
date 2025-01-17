import os

from itertools import product

import numpy as np

from msgwam import config

from .strategies import _get_stochastic_overrides, _get_integration
from .utils import get_rmse, load_data

_RESAMPLE = 6 * 3600

def save_coarsenings() -> None:
    """
    Integrate over the grid of vertical and spectral resolutions with low
    `config.n_max`, saving the output of each configuration.
    """

    overrides = _get_stochastic_overrides('1')
    for dr, n_source in product(*_get_grid()):
        overrides['dr_init'] = float(dr)
        overrides['n_source'] = n_source

        with config.override(**overrides):
            ds = _get_integration('stochastic').mean('sample')
            ds.to_netcdf(_get_path(dr, n_source))

def update_config(*names: str) -> None:
    """
    Update the values of `dr_init` and `n_source` in the specified configuration
    files to the best values found during the grid search.

    Parameters
    ----------
    names
        List of configuration names to update. If not passed, only the file for
        the currently loaded configuration will be changed, but other names can
        be included if multiple experiments should use the same coarsening.

    """

    drs, n_sources = _get_grid()
    errors = _get_normalized_errors()
    i, j = np.unravel_index(np.argmin(errors), errors.shape)

    if not names:
        names = [config.name]

    for name in names:
        with open(f'config/{name}.toml') as f:
            lines = f.readlines()

        with open(f'config/{name}.toml', 'w') as f:
            for line in lines:
                if line.startswith('dr_init'):
                    f.write(f'dr_init = {drs[i]}\n')

                elif line.startswith('n_source'):
                    f.write(f'n_source = {n_sources[j]}\n')

                else:
                    f.write(line)

def _get_error_profiles() -> np.ndarray:
    """
    Compute root-mean-square errors as a function of height for all coarsenings.

    Returns
    -------
    np.ndarray
        Two-dimensional grid of root-mean-square error profiles whose last
        dimension corresponds to height.

    """

    drs, n_sources = _get_grid()
    profiles = np.zeros((len(drs), len(n_sources), config.n_grid))
    ref = load_data('reference', resample=_RESAMPLE)

    for i, dr in enumerate(drs):
        for j, n_source in enumerate(n_sources):
            flux = load_data(_get_path(dr, n_source), resample=_RESAMPLE)
            profiles[i, j] = get_rmse(ref, flux)

    return profiles

def _get_path(dr: int, n_source: int) -> str:
    """
    Get the path where each integration output should be saved.

    Parameters
    ----------
    dr
        Value for `config.dr_init`.
    n_source
        Value for `config.n_source`.

    Returns
    -------
    str
        Path where integration data should be saved.

    """

    return f'data/{config.name}/coarsenings/dr-{dr}_n-source-{n_source}.nc'

def _get_grid() -> tuple[list[int], list[int]]:
    """
    Return lists of values for `config.dr_init` and `config.n_source` within
    which to search for the optimal coarse configuration.

    Returns
    -------
    list[int]
        Values for `config.dr_init`.
    list[int]
        Values for `config.n_source`.

    """

    drs = [500 * i for i in range(1, 11)]
    n_sources = [4 + 6 * i for i in range(10)]
    k = int(os.getenv('SLURM_ARRAY_TASK_ID', -1))

    if k > -1:
        drs = [drs[k // len(n_sources)]]
        n_sources = [n_sources[k % len(n_sources)]]

    return drs, n_sources[::-1]

def _get_normalized_errors() -> np.ndarray:
    """
    Get the normalized total error for each coarse configuration.

    Returns
    -------
    np.ndarray
        Two-dimensional grid of normalized errors averaged over all heights.

    """

    profiles = _get_error_profiles()
    ref = load_data('reference', resample=_RESAMPLE)
    rms = get_rmse(ref).values

    return (profiles / rms).mean(axis=-1)
