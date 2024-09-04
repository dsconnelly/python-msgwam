import sys

sys.path.insert(0, '.')
from msgwam import config
from msgwam.integration import SBDF2Integrator

N_MAX = 2502
SPEEDUP = 9

def integrate(strategy: str, mode: str) -> None:
    """
    Integrate the model with the loaded configuration file. For the stochastic
    strategy, performs multiple runs and saves the average.

    Parameters
    ----------
    strategy
        Name of the strategy to run. Must correspond to the name of one of the
        setup functions in strategies.py.
    mode
        Whether to run with a `'prescribed'` or `'interactive'` mean flow. Or,
        if set to `'convective'`, runs with prescribed mean winds but a source
        spectrum that varies in time according to ICON convection data.

    """
    
    func_name = '_' + strategy.replace('-', '_')
    globals()[func_name]()

    if mode == 'interactive':
        config.interactive_mean = True

    elif mode == 'convective':
        config.spectrum_type = 'from_file'

    config.refresh()
    ds = SBDF2Integrator().integrate()

    fname = strategy + f'-{mode}'
    ds.to_netcdf(f'data/{config.name}/{fname}.nc')
    config.reset()

def _reference() -> None:
    config.dt = 40
    config.dt_launch = 40
    config.n_ray_max = 150000
    config.n_increment = 10000

    config.n_source *= 5
    config.dr_init /= 5

def _many_fine() -> None:
    config.n_ray_max = N_MAX
    config.purge_mode = 'energy'

def _few_fine() -> None:
    config.n_ray_max = N_MAX // SPEEDUP
    config.purge_mode = 'energy'

def _many_coarse() -> None:
    config.n_ray_max = N_MAX
    config.purge_mode = 'energy'

    root = int(SPEEDUP ** 0.5)
    config.n_source //= root
    config.dr_init *= root

def _few_coarse() -> None:
    _many_coarse()
    config.n_ray_max = N_MAX // SPEEDUP

def _few_network() -> None:
    _few_coarse()
    config.source_type = 'network'

def _intermittent() -> None:
    _few_fine()
    config.dt_launch = 3 * 60 * 60
