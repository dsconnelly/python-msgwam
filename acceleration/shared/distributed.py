from itertools import product as _product
from os import environ, listdir, remove
from typing import Any, Iterable, Iterator

import numpy as np

from .. import hyperparameters as hp

N_TASKS = int(environ.get('SLURM_ARRAY_TASK_COUNT', 1))

def add_task_info(fname: str) -> str:
    """
    Given a file name, return a version of that string with information about
    the current task added just before the file suffix, so that distributed
    computations can easily save their results.

    Parameters
    ----------
    fname
        File name without task information.

    Returns
    -------
    str
        File name with added task information, unless there is only one task, in
        which case this function is a no-op.

    """

    if N_TASKS == 1:
        return fname
    
    *parts, suffix = fname.split('.')
    stem = '.'.join(parts)

    return stem + f'_task-{hp.task_id}' + f'.{suffix}'

def combine(path: str, remove_after: bool=False) -> None:
    """Combine arrays created by different SLURM tasks with a common prefix."""

    *parts, bname = path.split('/')
    bname, suffix = bname.split('.')
    dname = '/'.join(parts)

    paths = []
    for fname in listdir(dname):
        if fname.startswith(bname + '_'):
            paths.append(f'{dname}/{fname}')

    paths = sorted(paths)
    data = np.concatenate(list(map(np.load, paths)), axis=0)
    np.save(f'{dname}/{bname}.{suffix}', data)

    if remove_after:
        for path in paths:
            remove(path)

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
