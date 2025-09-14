import tomllib

from contextlib import contextmanager
from typing import Any, Iterator, Literal

import numpy as np

from .constants import ROT_EARTH

_DEFAULTS = {}

################################################################################
# global integration settings
################################################################################∂
propagator_type: Literal['instantaneous', 'network', 'transient']
source_type: Literal['constant', 'network', 'packet', 'stochastic']
spectrum_type: Literal['mima', 'custom']

################################################################################
# input and output
################################################################################
average_output: bool
prescribed_mean_file: str
spectrum_file: str

################################################################################
# time stepping
################################################################################
dt: int
dt_output: int
n_day: int

################################################################################
# mean state
################################################################################
H_rho: float
latitude : float
n_grid: int
N_ref: float
rho_ref: float
tau_nudge: float
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
dt_launch: int
equal_flux: bool
extrinsic: bool
flux_bc: float
n_source: int
r_source: float

################################################################################
# 'transient' propagator
################################################################################
cfl_mode: Literal['warn', 'raise']
dr_ghost: float
dr_max: float
dr_min: float
dr_source: float
jitter: float
logging: bool
max_age: int
max_dt_multiplier: int
max_overshoot: float
min_flux: float
n_chromatic: int
n_increment: int
n_max: int
n_sponge: int
prune_by: Literal[
    'cg_r',
    'energy',
    'flux',
    'importance',
    'none',
    'random'
]
prune_hours: int
shapiro_filter: bool

################################################################################
# 'network' propagator
################################################################################
lookback: int
n_history: int
network_path: str
time_horizon: float

################################################################################
# 'packet' source
################################################################################
n_repeat: int

################################################################################
# 'stochastic' source
################################################################################
epsilon: float

################################################################################
# 'mima' spectrum
################################################################################
flux_bc_ex: float
flux_bc_tr: float
cp_width_ex: float
cp_width_tr: float
lat_tropics: float
source_dlat: float
T_hat_source: float

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
    name = '.'.join(name.split('.')[:-1])
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

    for k, v in config.items():
        if not k.endswith('file'):
            continue

        data_dir = 'data/' + config['name']
        config[k] = v.replace('DATA_DIR', data_dir)

    if config['dt_launch'] < 0:
        config['dt_launch'] = config['dt']

    config['n_steps'] = int(86400 * config['n_day'] / config['dt']) + 1
    config['n_skip'] = round(config['dt_output'] / config['dt'])

    latitude = np.deg2rad(config['latitude'])
    config['f'] = 2 * ROT_EARTH * np.sin(latitude)

    if config['source_type'] != 'stochastic':
        config['epsilon'] = 1

    if isinstance(config['r_source'], str):
        def parse_line(line: str) -> np.ndarray:
            values = map(float, line.strip().split())
            return np.array(list(values))
        
        with open(config['r_source']) as f:
            lats, levels, *_ = map(parse_line, f)

        config['r_source'] = np.interp(config['latitude'], lats, levels)

def _is_valid(value: Any, annotation: Any) -> bool:
    """
    Check if a config setting is given its type annotation. Usually, this just
    amounts to an `isinstance` check, but this function defines the appropriate
    special behavior for annotations that do not support `isinstance`.

    Parameters
    ----------
    value
        Configuration value loaded from a file or supplied by the user.
    annotation
        Type annotation for the variable as read from `__annotations__`. Taken
        as a parameter directly so that the function can be used recursively.

    Returns
    -------
    bool
        Whether `value` is an appropriate value given the annotation for `key`
        defined in this module.
    
    """

    if 'Literal' in str(annotation):
        return value in annotation.__args__

    if annotation is float:
        return isinstance(value, (int, float))

    if 'list' in str(annotation):
        cls = annotation.__args__[0]
        return all(_is_valid(x, cls) for x in value)

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
        if not _is_valid(value, __annotations__[key]):
            raise ValueError(f'Invalid config setting: {key} = {value}')
        
    globals().update(config)
