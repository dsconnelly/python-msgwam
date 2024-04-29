import sys

import numpy as np

sys.path.insert(0, '.')
from msgwam import config
from msgwam.integration import SBDF2Integrator

def integrate(strategy: str) -> None:
    """
    Integrate the model with the loaded configuration file. For the stochastic
    strategy, performs multiple runs and saves the average.

    Parameters
    ----------
    strategy
        Name of the strategy to run. Must correspond to the name of one of the
        setup functions in strategies.py.

    """
    
    func_name = '_' + strategy.replace('-', '_')
    globals()[func_name]()
    config.refresh()

    n_trials = 10 if strategy == 'stochastic' else 1
    u, v, pmf_u, pmf_v = 0, 0, 0, 0

    for _ in range(n_trials):
        ds = SBDF2Integrator().integrate().to_dataset()

        u = u + ds['u'].values
        v = v + ds['v'].values
        pmf_u = pmf_u + ds['pmf_u'].values
        pmf_v = pmf_v + ds['pmf_v'].values

    ds['u'].values = u / n_trials
    ds['v'].values = v / n_trials
    ds['pmf_u'].values = pmf_u / n_trials
    ds['pmf_v'].values = pmf_v / n_trials

    ds.to_netcdf(f'data/{config.name}/{strategy}.nc')
    config.reset()

def _reference() -> None:
    pass

def _ICON() -> None:
    config.n_ray_max = 2500
    config.purge = True

def _do_nothing() -> None:
    config.n_ray_max = 250
    config.purge = True

def _coarse_square() -> None:
    n_source = int(config.n_source / np.sqrt(10))
    n_source = n_source + (n_source % 2)

    config.n_ray_max = 250
    config.n_source = n_source
    config.dr_init *= np.sqrt(10)
    config.purge = True

def _coarse_tall() -> None:
    config.n_ray_max = 250
    config.dr_init *= 10
    config.purge = True

def _coarse_wide() -> None:
    n_source = int(config.n_source / 10)
    n_source = n_source + (n_source % 2)

    config.n_ray_max = 250
    config.n_source = n_source
    config.purge = True

def _stochastic() -> None:
    config.source_type = 'stochastic'
    config.n_ray_max = 250
    config.epsilon = 0.1
    config.purge = True