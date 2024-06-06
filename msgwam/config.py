import tomllib

from typing import Optional

import torch

from .constants import ROT_EARTH

_DEFAULTS = None

def load(path: str) -> None:
    """
    Load configuration file from a TOML file and update the module namespace so
    that parameters can be accessed as `config.foo`. Stores the loaded config
    settings in a global variable, so that they can be reverted to later. Also
    determines and saves the name of the config setup, since we can only do so
    at this stage when we have access to the config file path.
    
    """

    global _DEFAULTS
    with open(path, 'rb') as f:
        _DEFAULTS = tomllib.load(f)

    name = path.split('/')[-1]
    name = name.split('.')[0]
    _DEFAULTS['name'] = name

    refresh(_DEFAULTS.copy())

def refresh(config: Optional[dict[str]]=None) -> None:
    """
    Some parameters accessible through the config module are not specified in
    the config file, but instead derived automatically. This function adds those
    derived parameters to the namespace, and is available as a separate function
    in case the user wishes to change config parameters and update the derived
    values without loading a new config file.

    Parameters
    ----------
    config
        Dictionary of config parameters to add to the module namespace alongside
        derived parameters. If None, the existing module namespace will be used.

    """

    if config is None:
        config = globals().copy()

    config['latitude'] = torch.deg2rad(torch.as_tensor(config['latitude']))
    config['f0'] = 2 * ROT_EARTH * torch.sin(config['latitude'])

    config['n_t_max'] = int(86400 * config['n_day'] / config['dt']) + 1
    config['n_skip'] = round(config['dt_output'] / config['dt'])

    config['r_ghost'] = config['r_launch'] - config['dr_init']

    if 'coarse_height' in config and 'coarse_width' in config:
        height, width = config['coarse_height'], config['coarse_width']
        config['rays_per_packet'] = height * width

    globals().update(config)

def reset() -> None:
    """
    Reset the config parameters to the values from the most recently-read config
    file. This function exists so that the config settings can be changed and
    then reverted without having to read in a file again.

    """

    refresh(_DEFAULTS.copy())