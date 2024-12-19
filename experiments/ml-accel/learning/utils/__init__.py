from hashlib import sha1

from .bases import apply_basis, parse_proxies
from .distributed import add_task_info, combine_data, get_workload
from .io import get_indices, load_data
from .overrides import get_overrides, with_overrides
from .statistics import standardize

__all__ = [
    'add_task_info',
    'apply_basis',
    'combine_data',
    'get_indices',
    'get_overrides',
    'get_workload',
    'load_data',
    'make_seed',
    'parse_proxies',
    'standardize',
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
