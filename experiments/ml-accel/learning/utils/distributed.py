from os import environ, listdir

import numpy as np
import torch

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

def combine_data(path: str) -> None:
    """
    Once data has been generated and saved to disk by separate processes, this
    function can be called to combine those files and save the result.

    Parameters
    ----------
    path
        String where the combined data should be saved. It is assumed that there
        are one or more arrays or tensors saved with the same path except with
        task info added as by `add_task_info`.

    """

    parts, base_name = path.split('/')
    dir_name = '/'.join(parts)

    is_valid = lambda s: s.startswith(base_name + '_task-')
    fnames = sorted(filter(is_valid, listdir(dir_name)))

    if len(fnames) == 0:
        return

    suffix = base_name.split('.')[-1]
    lib = {'npy' : np, 'pkl' : torch}[suffix]
    data = lib.vstack(map(lib.load, fnames))

    if suffix == 'npy':
        np.save(path, data)

    elif suffix == 'pkl':
        torch.save(data, path)

def get_workload(n_data: int) -> tuple[int, int]:
    """
    Given a the size of the data that needs to be operated on, get the start and
    end indices of the chunk of data that should be worked on by this task.

    Parameters
    ----------
    n_data
        Size of the dataset being worked on.

    Returns
    -------
    int, int
        Start (inclusive) and end (exclusive) indices into the dataset.

    """

    step, rem = divmod(n_data, N_TASKS)
    start = hp.task_id * step + min(hp.task_id, rem)
    end = start + step + int(hp.task_id < rem)

    return start, end
