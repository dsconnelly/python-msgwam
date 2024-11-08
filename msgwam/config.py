import tomllib

from contextlib import contextmanager
from typing import Any, Iterator, Literal

import numpy as np

from .constants import ROT_EARTH

_DEFAULTS = {}

################################################################################
# global integration settings
################################################################################
mean_state_type: Literal['interactive', 'prescribed']
propagator_type: Literal['network', 'steady', 'transient']
source_type: Literal['deterministic', 'network', 'stochastic']
spectrum_type: Literal['convective', 'custom', 'gaussians']

################################################################################
# input and output
################################################################################
average_output: bool
prescribed_wind_file: str

################################################################################
# time stepping
################################################################################
dt: int
dt_output: int
n_day: int

################################################################################
# mean state
################################################################################
boussinesq: bool
H_rho: float
n_grid: int
N_ref: float
rho_ref: float
z_max: float
z_min: float

################################################################################
# viscosity and dissipation
################################################################################
dissipation: float
mu: float

################################################################################
# gravity wave source and spectrum
################################################################################
c_max: float
n_source: int

################################################################################
# 'transient' propagator
################################################################################
check_sign_changes: bool
dr_init: float
min_flux: float
n_chromatic: int
n_increment: int
n_max: int
prune_by: Literal['energy', 'none', 'random']
shapiro_filter: bool

################################################################################
# 'stochastic' source
################################################################################
epsilon: float

################################################################################
# 'gaussians' spectrum
################################################################################
flux_bc: float
c_center: float
c_width: float
dk_init: float
direction: float
dl_init: float
period_hours: float

################################################################################
# derived settings
################################################################################
f: float
name: str
n_skip: int
n_steps: int

def load(path: str) -> None:
    """
    Load configuration data from a TOML file and update the module namespace so
    that parameters can be accessed as `config.foo`. Stores the loaded config
    settings in a global variable, so that they can be reverted to later if need
    be. Also determines and saves the name of the config setup, since we can do
    so only at this stage when we have access to the config file path.

    """

    global _DEFAULTS
    with open(path, 'rb') as f:
        _DEFAULTS = tomllib.load(f)

    name = path.split('/')[-1]
    name = name.split('.')[0]
    _DEFAULTS['name'] = name

    _update(_DEFAULTS)

@contextmanager
def override(**kwargs) -> Iterator[None]:
    """
    Context manager to allow the values in `kwargs` to be used as configuration
    settings for the duration of the `with` block. After the TOML namelist is
    loaded at initialization, configuration settings should only be modified by
    means of this function.

    Note that this function is type-hinted as an iterator since `contextlib`
    requires a generator, but there are no useful yield or return values.

    Parameters
    ----------
    kwargs
        Pairs of configuration keys and values to temporarily override.

    """

    _update(kwargs)

    try:
        yield

    finally:
        _update(_DEFAULTS)

def _add_derived(config: dict[str, Any]) -> None:
    """
    Get the configuration settings which are calculated internally instead of
    being supplied by the user.

    Parameters
    ----------
    config
        Dictionary of configuration settings, as supplied by `_update`.
    
    """

    config['n_steps'] = int(86400 * config['n_day'] / config['dt']) + 1
    config['n_skip'] = round(config['dt_output'] / config['dt'])

    latitude = np.deg2rad(config.pop('latitude'))
    config['f'] = 2 * ROT_EARTH * np.sin(latitude)

def _is_valid(key: str, value: Any) -> bool:
    """
    Check if a config setting is given its type annotation. Usually, this just
    amounts to an `isinstance` check, but this function defines the appropriate
    special behavior for annotations that do not support `isinstance`.

    Parameters
    ----------
    key
        Name of configuration setting to validate.
    value
        Configuration value loaded from a file or supplied by the user.

    Returns
    -------
    bool
        Whether `value` is an appropriate value given the annotation for `key`
        defined in this module.
    
    """

    annotation = __annotations__[key]

    if 'Literal' in str(annotation):
        return value in annotation.__args__
    
    if annotation is float:
        return isinstance(value, (int, float))
    
    return isinstance(value, annotation)

def _update(config: dict[str, Any]) -> None:
    """
    Validate the provided configuration settings, derive the internal settings
    from those provided, and updatae the module namespace. This function should
    be used whenever configuration settings are changed to avoid invalid states.

    Parameters
    ----------
    config
        Dictionary of config parameters to add to the module namespace.

    """

    config = dict(_DEFAULTS, **config)
    _add_derived(config)

    for key, value in config.items():
        if not _is_valid(key, value):
            raise ValueError(f'Invalid config setting: {key} = {value}')
        
    globals().update(config)
