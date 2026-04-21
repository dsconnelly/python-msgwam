from __future__ import annotations
from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from msgwam import config
from msgwam.integration import integrate

from ..hyperparameters import strategies as hp

from .overrides import get_overrides

if TYPE_CHECKING:
    from msgwam.means import MeanState
    from msgwam.propagators import TransientPropagator

def get_integration() -> xr.Dataset:
    """
    Integrate the solver and return a dataset containing the outputs. Broken out
    as a separate function so that other code can perform integrations without
    saving results to disk.

    Returns
    -------
    xr.Dataset
        Dataset containing the integration outputs. Will also contain an extra
        dimension `'member'` which either ranges over ensemble members, for
        configurations with some randomness, or is a singleton.

    """

    includes = ['intermittent', 'stochastic']
    func = lambda i: integrate().assign_coords(member=i)
    ensemble = (config.jitter > 0) or (config.source_type in includes)
    datasets = map(func, range(hp.n_ensemble if ensemble else 1))

    return xr.concat(datasets, dim='member')

def save_strategy(strategy: str, *args: str) -> None:
    """
    Integrate with the configurations specific to the given strategy and save
    result to disk as a netCDF file. This is the function that is meant to be
    called from the command line.

    Parameters
    ----------
    strategy
        Name of the configuration strategy with which to integrate.
    args
        Arguments to pass to the override function, if any.

    """

    with config.override(**get_overrides(strategy, *args)):
        ds = get_integration()

    fname = '-'.join([strategy, *map(str, args)])
    base = f'data/{config.name}/strategies/{fname}'
    ds.mean('member').to_netcdf(f'{base}.nc')

    if len(ds['member']) > 1:
        ds.to_netcdf(f'{base}-ensemble.nc')

def save_trajectories(strategy: str, *args: str) -> None:
    """
    Integrate with a given strategy and also save ray volume trajectory data.
    Provided as a standalone function since saving ray volume data is rather
    and expensive and likely only needed for illustrative purposes.
    """

    idx = defaultdict(lambda : 0)
    n_hist = 5 * 86400 // config.dt + 1
    factory = lambda: np.nan * np.zeros((6, n_hist))
    data = defaultdict(factory)
    k, l = {}, {}

    def callback(
        mean: MeanState,
        prop: TransientPropagator,
        _
    ) -> None:
        """Callback function to extract ray volume properties."""

        wvn = prop.k + prop.l
        cg = prop._get_cg_r(mean)
        omega_hat = prop._get_omega_hat(mean)

        cp_hat = omega_hat / wvn
        energy = abs(prop.action * omega_hat)
        flux = abs(wvn * prop.action * cg)

        to_log = prop._valid & (prop.age < 5 * 86400)
        to_log, = np.where(to_log)

        for j, m in zip(to_log, prop.meta[to_log].astype(int)):
            data[m][:, idx[m]] = [
                prop.r[j],
                prop.dr[j],
                cp_hat[j],
                cg[j],
                energy[j] * prop.dr[j],
                flux[j] * prop.dr[j]
            ]

            idx[m] = idx[m] + 1
            k[m] = prop.k[j]
            l[m] = prop.k[j]

    with config.override(**get_overrides(strategy, *args)):
        _ = integrate(callback)

    kwargs = {
        'meta' : np.array(list(data.keys())),
        'age' : np.arange(n_hist) * config.dt,
        'k' : ('meta', np.array(list(k.values()))),
        'l' : ('meta', np.array(list(l.values())))
    }

    stacked = np.stack(list(data.values()), axis=0)
    for i, name in enumerate(['r', 'dr', 'cp_hat', 'cg', 'energy', 'flux']):
        kwargs[name] = (('meta', 'age'), stacked[:, i])

    fname = '-'.join([strategy, *args]) + '-trajectories.nc'
    xr.Dataset(kwargs).to_netcdf(f'data/{config.name}/strategies/{fname}')
    