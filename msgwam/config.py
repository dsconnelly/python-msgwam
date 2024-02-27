import numpy as np
import tomllib

from .constants import ROT_EARTH

def load_config(path: str) -> None:
    """
    Load configuration data from a TOML file and update the module namespace so
    that parameters can be accessed as `config.{name}`. Also perform simple
    precalculations of values that are derived from configuration data but used
    in several places.

    Parameters
    ----------
    path
        Path to TOML configuration file.
        
    """

    with open(path, 'rb') as f:
        config = tomllib.load(f)
        
    refresh(config)

def refresh(config: dict[str]=None) -> None:
    """
    Calculate internal parameters and update the module namespace. Available as
    a separate function in case the user wishes to change parameters without
    reloading a config file.

    Parameters
    ----------
    config
        Dictionary of config parameters. If None, the global module namespace
        will be used instead.

    """

    if config is None:
        config = globals().copy()

    config['latitude'] = np.deg2rad(config['latitude'])
    config['f0'] = 2 * ROT_EARTH * np.sin(config['latitude'])

    config['n_t_max'] = int(86400 * config['n_day'] / config['dt']) + 1
    config['n_skip'] = round(config['dt_output'] / config['dt'])

    if 'r_launch' in config:
        config['r_ghost'] = config['r_launch'] - config['dr_init']

    globals().update(config)    
