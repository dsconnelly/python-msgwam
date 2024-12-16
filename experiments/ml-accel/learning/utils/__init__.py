from .bases import apply_basis
from .distributed import add_task_info, combine_data, get_workload
from .io import get_indices, load_data, load_model
from .overrides import get_overrides, with_overrides

__all__ = [
    'add_task_info',
    'apply_basis',
    'combine_data',
    'get_indices',
    'get_overrides',
    'get_workload',
    'load_data',
    'load_model',
    'make_seed',
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
    return hash(to_hash) % 2 ** 32