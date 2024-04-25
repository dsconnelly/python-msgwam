import numpy as np
import tomllib

from .constants import ROT_EARTH

_DEFAULTS = None

def load_config(path: str) -> None:
    """
    Load configuration data from a TOML file and update the module namespace so
    that parameters can be accessed as `config.{name}`. Stores the loaded config
    settings in a global variable, so that they can be reverted to later. Also
    determines and saves the name of the config setup, since we can only do that
    when we have access to the config file path.

    Parameters
    ----------
    path
        Path to TOML configuration file.
        
    """

    global _DEFAULTS
    with open(path, 'rb') as f:
        _DEFAULTS = tomllib.load(f)

    name = path.split('/')[-1]
    name = name.split('.')[0]
    _DEFAULTS['name'] = name

    refresh(_DEFAULTS.copy())

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

def reset() -> None:
    """
    Reset the config state to the values from the most recently-read config
    file. Exists so that the config settings can be changed and then reverted
    without having to reload a file.

    """

    refresh(_DEFAULTS.copy())