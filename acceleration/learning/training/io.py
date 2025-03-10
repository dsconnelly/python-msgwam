import os

from itertools import cycle
from typing import Iterator, Literal, Optional

import numpy as np
import torch

from torch.utils.data import DataLoader, Dataset, Sampler

from msgwam import config

from ...hyperparameters import training as hp

_Phase = Literal[1, 2, 3]

def get_loader(phase: _Phase, subsets: list[str]) -> DataLoader:
    """
    Get a `DataLoader` that properly samples from training data saved across
    multiple files on disk.

    Parameters
    ----------
    phase, subsets
        Arguments to pass to `_MultifileDataset`.

    Returns
    -------
    DataLoader
        Loader sampling from multiple files with smart caching.

    """

    ds = _MultifileDataset(phase, subsets)
    return DataLoader(ds, hp.batch_size, sampler=_MultifileSampler(ds))

class _MultifileDataset(Dataset):

    def __init__(self, phase: _Phase, subsets: list[str]) -> None:
        """
        At initialization, the file names in the data directory are parsed to
        determine the number of available tasks, and the size of the dataset is
        stored accordingly.

        Parameters
        ----------
        phase
            What phase of training this dataset is to be used for. Will affect
            which files are treated as inputs and which as targets.
 
        """

        self._data_dir = f'data/{config.name}/training'
        parse = lambda s: int(s.split('.')[0].split('-')[-1])
        n_tasks = max(map(parse, os.listdir(self._data_dir))) + 1

        a = int(0.7 * n_tasks)
        b = a + int(0.15 * n_tasks)
        idx = list(range(n_tasks))

        self._tasks = []
        for s in subsets:
            start, end = {'tr' : (0, a), 'va' : (a, b), 'te' : (b, n_tasks)}[s]
            self._tasks = self._tasks + idx[start:end]

        k = self._tasks[0]
        data: np.ndarray = np.load(f'{self._data_dir}/S_task-{k}.npy')
        self.n_per_task = data.shape[0] + (phase == 1)
        self._phase = phase

        self._i: Optional[int] = None
        self._cache: Optional[list[torch.Tensor]] = None

    def __getitem__(self, idx: tuple[int, int]) -> tuple[torch.Tensor, ...]:
        """
        Samples are retrieved by first checking whether the requested file is
        already loaded, and loading that file if not. Then the requested samples
        within that file are returned for each input and target tensor.

        Parameters
        ----------
        idx
            Tuple whose first entry corresponds to the task file to be loaded
            and whose second entry indicates the rows within that file.

        Returns
        -------
        tuple[torch.Tensor, ...]
            Input and target tensors. The last tensor is the target data.

        """

        i, j = idx
        if i != self._i:
            self._update_cache(i)

        return tuple([a[j] for a in self._cache])

    def __len__(self) -> int:
        """
        The number of samples in the dataset is the number of task files found
        times the number of samples in each file.

        Returns
        -------
        int
            Number of samples across all files.

        """

        return self.n_tasks * self.n_per_task
    
    @property
    def n_tasks(self) -> int:
        """
        At initialization, the dataset records the task file numbers that should
        load depending on whether the dataset holds training, validation, or
        test data. The number of task files is just the length of that list.
        """

        return len(self._tasks)

    def _update_cache(self, i: int) -> None:
        """
        Load input and output files corresponding to the given task. Which files
        are treated as inputs and outputs is determined by `self._phase`. The
        loaded files are cached as instance attributes, as is the index `i`.

        Parameters
        ----------
        i
            Integer corresponding to one of the tasks during generation.

        """

        print(f'caching {i}')

        inputs = ['R'] if self._phase == 1 else ['u', 'S', 'R']
        Xs = list(map(self._load, inputs, cycle([i])))
        output = 'R' if self._phase == 2 else 'F'
        Y = self._load(output, i)

        if self._phase != 1:
            Xs[-1] = Xs[-1][:-1]
            Y = Y[1:]

        self._i = i
        self._cache = Xs + [Y]

    def _load(self, kind: str, i: int) -> torch.Tensor:
        """
        Load a particular kind of data from a given task file. Builds the right
        path and handles conversion to `Tensor`.

        Parameters
        ----------
        kind
            What kind of data to load. Must be `'u'`, `'S'`, `'R'`, or `'F'`.
        i
            Task file to load from.
        
        Returns
        -------
        torch.Tensor
            Loaded tensor.

        """

        path = f'{self._data_dir}/{kind}_task-{self._tasks[i]}.npy'
        data = torch.as_tensor(np.load(path))

        if kind == 'F':
            data = data.sum(dim=1)

        return data

class _MultifileSampler(Sampler):

    def __init__(
        self,
        dataset: _MultifileDataset,
        seed: Optional[int]=123
    ) -> None:
        """
        Save the number of files and objects per task to be iterated over later.
        Also construct a random number generator for reproducibility.

        Parameters
        ----------
        dataset
            Initialized `MultifileDataset` used to build the iterator.
        seed
            Seed for the random number generator.

        """

        self._i_max = dataset.n_tasks
        self._j_max = dataset.n_per_task
        self._rng = np.random.default_rng(seed)

    def __iter__(self) -> Iterator[tuple[int, int]]:
        """
        Iterate over samples in the dataset, using all samples in the current
        file before moving on to the next.
        """

        for i in self._rng.permutation(self._i_max):
            for j in self._rng.permutation(self._j_max):
                yield i, j

    def __len__(self) -> int:
        """The length is calculated as in `MultifileDatset`."""

        return self._i_max * self._j_max
