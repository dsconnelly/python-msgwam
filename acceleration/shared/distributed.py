from itertools import product as _product
from os import environ
from typing import Any, Iterable, Iterator

from .. import hyperparameters as hp

N_TASKS = int(environ.get('SLURM_ARRAY_TASK_COUNT', 1))

def get_workload(n: int) -> tuple[int, int]:
    """
    Given the total size of the data on which a calculation needs to be run, get
    the start and end indices of the chunk of data this task should work on.

    Parameters
    ----------
    n
        Total size of the dataset.

    Returns
    -------
    int, int
        Start (inclusive) and end (exclusive) indices into the dataset.

    """

    step, rem = divmod(n, N_TASKS)
    start = hp.task_id * step + min(hp.task_id, rem)
    end = start + step + int(hp.task_id < rem)

    return start, end

def product(*args: Iterable[Any]) -> Iterator[tuple[Any, ...]]:
    """
    Iterate over the Cartesian product of a set of iterables, yielding only
    those combinations that should be handled by this task. If there is only one
    task, this function is equivalent to `itertools.product`.

    Parameters
    ----------
    args
        Iterables of which to take the Cartesian product.

    Yields
    ------
    tuple[Any, ...]
        Element of the Cartesian product of args.

    """

    n = 1
    for arg in args:
        n = n * len(arg)

    start, end = get_workload(n)
    for i, values in enumerate(_product(*args)):
        if start <= i < end:
            yield values
