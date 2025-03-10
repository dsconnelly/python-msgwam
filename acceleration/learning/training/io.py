import os

from itertools import cycle
from typing import Iterator, Optional

import numpy as np
import torch

from torch.utils.data import DataLoader, Dataset, Sampler

from msgwam import config

from ...hyperparameters import training as hp

def get_loader(
    kinds: str | list[str],
    subsets: str | list[str],
    batch_size: Optional[int]=None
) -> DataLoader:
    """
    Get a `DataLoader` that properly samples from training data saved across
    multiple files on disk.

    Parameters
    ----------
    kinds
        What kind of data to load. Available kinds are `'u'`, `'S'`, `'R'`,
        and `'F'`, corresponding to mean wind, source activity, ray volume
        state, and momentum flux profiles, respectively. In addition, `'R'`
        and `'F'` can be suffixed with `'i'` or `'o'` in training tasks
        where inputs and outputs should be offset.
    subsets
        Which subsets to load. Must be `'tr'`, `'va'`, or `'te'`.

    Returns
    -------
    DataLoader
        Loader sampling from multiple files with smart caching.

    """

    if not isinstance(kinds, list):
        kinds = [kinds]

    if not isinstance(subsets, list):
        subsets = [subsets]

    if batch_size is None:
        batch_size = hp.batch_size

    ds = _MultifileDataset(kinds, subsets)
    return DataLoader(ds, batch_size, sampler=_MultifileSampler(ds))

class _MultifileDataset(Dataset):
    """
    Internal class for loading and caching training data saved across multiple
    files. Should be used in concert with `_MultifileSampler`.
    """

    def __init__(self, kinds: list[str], subsets: list[str]) -> None:
        """
        At initialization, the file names in the data directory are parsed to
        determine the number of available chunks, and the size of the dataset is
        stored accordingly.

        Parameters
        ----------
        kinds, subsets
            Parameters as passed to `get_loader`.

        """

        self._i = -1
        self._cache: list[torch.Tensor] = []
        self._kinds = kinds

        self._data_dir = f'data/{config.name}/training'
        parse = lambda s: int(s.split('.')[0].split('-')[-1])
        n_chunks = max(map(parse, os.listdir(self._data_dir))) + 1

        a = min(int(0.7 * n_chunks), n_chunks - 2)
        b = a + max(int(0.15 * n_chunks), 1)
        idx = list(range(n_chunks))

        self._chunks = []
        for s in subsets:
            start, end = {'tr' : (0, a), 'va' : (a, b), 'te' : (b, n_chunks)}[s]
            self._chunks = self._chunks + idx[start:end]

        k = self._chunks[0]
        data: np.ndarray = np.load(f'{self._data_dir}/S_task-{k}.npy')
        is_offset = any([len(kind) > 1 for kind in self._kinds])
        self.n_per_chunk = data.shape[0] + (not is_offset)

    def __getitem__(self, idx: tuple[int, int]) -> tuple[torch.Tensor, ...]:
        """
        Samples are retrieved by first checking whether the requested file is
        already cached, and caching that file if not. Then the requested samples
        within that file are returned for each input and target tensor.

        Parameters
        ----------
        idx
            Tuple whose first entry corresponds to the chunk to be loaded and
            whose second entry indicates rows within that chunk. Note that the
            chunk is given by index, so that it always starts at zero.

        Returns
        -------
        tuple[torch.Tensor, ...]
            Requsted samples.

        """

        i, j = idx
        if i != self._i:
            self._update_cache(i)

        return tuple([a[j] for a in self._cache])
    
    def __len__(self) -> int:
        """
        The number of samples in the dataset is the number of chunk files found
        times the number of samples in each chunk.

        Returns
        -------
        int
            Number of samples across all chunks.

        """

        return self.n_chunks * self.n_per_chunk

    @property
    def n_chunks(self) -> int:
        """
        At initialization, the dataset records the chunk numbers that should be
        loaded depending on whether it is to hold training, validation or test
        data. The number of chunks is just the length of that list.
        """

        return len(self._chunks)
    
    def _load(self, kind: str, i: int) -> torch.Tensor:
        """
        Load a particular kind of data from a given task file. Builds the right
        path and handles conversion to `Tensor`.

        Parameters
        ----------
        kind
            What kind of data to load, as passed to `__init__`.
        i
            Index of the chunk to load.
        
        Returns
        -------
        torch.Tensor
            Loaded tensor.

        """

        try:
            kind, shift = kind
            start = {'i' : None, 'o' : 1}[shift]
            end = {'i' : -1, 'o' : None}[shift]

        except ValueError:
            start, end = None, None

        path = f'{self._data_dir}/{kind}_task-{self._chunks[i]}.npy'
        data = torch.as_tensor(np.load(path))[slice(start, end)]

        if kind == 'F':
            data = data.sum(dim=1)

        return data

    def _update_cache(self, i: int) -> None:
        """
        Load files corresponding to the given chunk. The loaded files are cached
        as instance attributes, as is the currently cached index.

        Parameters
        ----------
        i
            Integer indexing one of the chunks produced during generation.

        """

        self._cache = list(map(self._load, self._kinds, cycle([i])))
        self._i = i

class _MultifileSampler(Sampler):
    """Internal class for sampling from a `_MultifileDataset`."""

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
            Initialized `_MultifileDataset` used to build the iterator.
        seed
            Seed for the random number generator.

        """

        self._i_max = dataset.n_chunks
        self._j_max = dataset.n_per_chunk
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
