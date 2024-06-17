import sys

sys.path.insert(0, '.')
from msgwam import config
from msgwam.integration import SBDF2Integrator

N_MAX = 2502
SPEEDUP = 9

def integrate(strategy: str, *args) -> None:
    """
    Integrate the model with the loaded configuration file. For the stochastic
    strategy, performs multiple runs and saves the average.

    Parameters
    ----------
    strategy
        Name of the strategy to run. Must correspond to the name of one of the
        setup functions in strategies.py.
    args
        Other arguments that might be passed to the setup function.

    """
    
    func_name = '_' + strategy.replace('-', '_')
    globals()[func_name](*args)
    config.refresh()

    n_trials = SPEEDUP if strategy == 'stochastic' else 1
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

    fname = strategy + ('-'.join(['', *args]) if args else '')
    ds.to_netcdf(f'data/{config.name}/{fname}.nc')
    config.reset()

def _reference() -> None:
    config.min_pmf = 1e-9

def _ICON(mode: str='energy') -> None:
    config.n_ray_max = N_MAX
    config.purge_mode = mode

def _do_nothing() -> None:
    config.n_ray_max = N_MAX // SPEEDUP
    config.purge_mode = 'energy'

def _coarse_square(mode: str='energy') -> None:
    config.source_type = 'coarse'
    _do_nothing()
    config.purge_mode = mode

    root = int(SPEEDUP ** 0.5)
    config.coarse_height = root
    config.coarse_width = root

def _coarse_tall() -> None:
    config.source_type = 'coarse'
    _do_nothing()

    config.coarse_height = SPEEDUP
    config.coarse_width = 1

def _coarse_wide() -> None:
    config.source_type = 'coarse'
    _do_nothing()

    config.coarse_height = 1
    config.coarse_width = SPEEDUP

def _stochastic() -> None:
    config.source_type = 'stochastic'
    config.epsilon = 1 / SPEEDUP
    _do_nothing()

def _network() -> None:
    _coarse_square()
    config.source_type = 'network'
