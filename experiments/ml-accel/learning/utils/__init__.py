from hashlib import sha1

from .distributed import (
    add_task_info,
    combine_data,
    get_generation_mode,
    get_workload
)
from .io import load_data
from .overrides import get_overrides, with_overrides
from .proxies import apply_basis, init_proxies, transform_proxies

__all__ = [
    'add_task_info',
    'apply_basis',
    'combine_data',
    'get_generation_mode',
    'get_overrides',
    'get_workload',
    'init_proxies',
    'load_data',
    'make_seed',
    'transform_proxies',
    'with_overrides'
]

def make_seed(*args) -> int:
    """
    Make a random seed by hashing a string composed of the given arguments.

    Parameters
    ----------
    args
        List of objects to use in generating the string. Must support conversion
        to string in some way or another.

    Returns
    -------
    int
        Appropriately bounded seed.

    """

    to_hash = ''.join(map(str, args))
    hashed = sha1(to_hash.encode()).digest()

    return int.from_bytes(hashed, 'big') % 2 ** 32
