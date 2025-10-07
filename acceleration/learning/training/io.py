from typing import Literal, Optional
from warnings import warn

import numpy as np
import torch
import xarray as xr

from torch.utils.data import DataLoader, TensorDataset

from ... import hyperparameters as hp

from .transforms import apply_smoothing

_SITES_TR = [
    'anchorage',
    'new-york',
    'lisbon',
    'miami',
    'maldives',
    'brisbane',
    'buenos-aires',
    'weddell-sea'
]

_SITES_TE = [
    'copenhagen',
    'singapore',
    'perth',
    'amundsen-sea'
]

def get_loaders(
    batch_size_tr: int,
    idx_tr: torch.Tensor,
    idx_ev: torch.Tensor,
    *args: torch.Tensor
) -> tuple[DataLoader, DataLoader]:
    """
    Package the training and evaluation data into `DataLoader` instances.

    Parameters
    ----------
    batch_size_tr
        Batch size to use for the training set.
    idx_tr, idx_ev
        Tensors that partition the data into training and evaluation sets.
    args
        Tensors to split and put into `DataLoader` instances.
        
    Returns
    -------
    DataLoader, DataLoader
        Loaders for the training and evaluation sets.
    
    """

    loaders = []
    for i, idx in enumerate([idx_tr, idx_ev]):
        ds = TensorDataset(*[arg[idx] for arg in args])
        batch_size = (1 - i) * batch_size_tr + i * 4096
        loaders.append(DataLoader(ds, batch_size, i == 0))

    return tuple(loaders)

def get_split(
    n_samples: int,
    eval_type: Literal['va', 'te'],
    seed: int=1234
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Get index arrays that can be used to split the data into subsets for
    training and evaluation.

    Parameters
    ----------
    n_samples
        How many total datapoints are available.
    eval_type
        Whether to use validation or test data as the evaluation set. If test
        data is used, the evaluation set will be pulled from MiMA scenarios that
        are entirely left out of the training data.
    seed
        Seed to use for random splitting if `eval_type == 'va'`.

    Returns
    -------
    torch.Tensor, torch.Tensor
        Indices for training and evaluation sets, respectively.

    """

    g = torch.Generator()
    g.manual_seed(seed)

    if eval_type == 'va':
        c = int(0.8 * n_samples)
        idx = torch.randperm(n_samples, generator=g)

    elif eval_type == 'te':
        total = len(_SITES_TR + _SITES_TE)
        c = (n_samples * len(_SITES_TR)) // total
        idx = torch.arange(n_samples)

    else:
        raise ValueError(f'Invalid eval_type: {eval_type}')

    return idx[:c], idx[c:]

def load_tensors(
    eval_type: Literal['va', 'te'],
    min_samples: Optional[int]=None,
    cached: bool=False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load input and target data from netCDF files saved to disk.

    Parameters
    ----------
    eval_type
        String specifying whether evaluation data should be from the validation
        set or the test set, used here to determine which files to read.

    Returns
    -------
    Tensor, Tensor, Tensor, Tensor
        Tensors of bulk momentum, bulk group velocity, mean wind, and target
        data for each sample, respectively. The appropriate component of the
        mean wind is selected, and it is negated for negative wavenumbers. The
        target data are the bulk momentum and velocity profiles at the next time
        step concatenated. The input and output bulk momentum profiles at each
        time step are normalized by the appropriate budget.

    """

    if cached:
        _make_path = lambda s: f'data/ml-accel/cached/{s}-{eval_type}.pkl'
        _load = lambda s: torch.load(_make_path(s), weights_only=True)
        outputs = map(_load, ['windN', 'M_in', 'M_out', 'D'])

        return tuple(outputs)

    base = 'data/ml-accel/training'
    sites = _SITES_TR + _SITES_TE * (eval_type == 'te')
    paths = [f'{base}/{site}-{i}.nc' for site in sites for i in range(1, 13)]
    args = [[], [], [], []]
    
    total = 0
    for path in paths:
        with xr.open_dataset(path) as ds:
            u = torch.as_tensor(ds['u'].values)
            v = torch.as_tensor(ds['v'].values)
            N = torch.as_tensor(ds['N'].values)
            lat = ds.attrs['latitude']

            M = ds['M_bulk'].values
            S = ds['source'].values
            D = ds['sink'].values

            for _ in range(hp.training.n_smoothing):
                M = apply_smoothing(M)
                S = apply_smoothing(S)
                D = apply_smoothing(D)

            M = torch.as_tensor(M)
            S = torch.as_tensor(S)
            D = torch.as_tensor(D)

        windN = _make_windN(u, v, N)
        lats = lat * torch.ones(windN.shape[0])
        windN = torch.hstack((windN, lats[:, None]))
        M_in, M_out, D = _make_pairs(M, S, D)

        args[0].append(windN)
        args[1].append(M_in)
        args[2].append(M_out)
        args[3].append(D)

        total = total + M_in.shape[0]
        if min_samples is not None and total >= min_samples:
            break

    outputs = tuple(torch.cat(arg, dim=0) for arg in args)
    for data, name in zip(outputs, ['windN', 'M_in', 'M_out', 'D']):
        torch.save(data, f'data/ml-accel/cached/{name}-{eval_type}.pkl')

    return outputs

def _make_windN(
    u: torch.Tensor,
    v: torch.Tensor,
    N: torch.Tensor
) -> torch.Tensor:
    """
    Get the appropriate component of the mean wind at each sample, and negate
    the wind profile for samples with negative wavenumber

    Parameters
    ----------
    u, v
        Zonal and meridional components of the mean wind, respectively.
    N
        Buoyancy frequency at each time step

    Returns
    -------
    torch.Tensor
        Wind profile to use in predicting each sample concatenated with N.

    """

    quad = torch.arange(4)[None, :, None]
    u, v, N = u[:, None], v[:, None], N[:, None]
    u, v, N, quad = torch.broadcast_tensors(u, v, N, quad)

    is_zonal = (torch.remainder(quad, 2) == 0).int()
    wind = is_zonal * u + (1 - is_zonal) * v
    wind[quad > 1] = -wind[quad > 1]

    wind = wind[:-1].flatten(0, 1)
    N = N[:-1].flatten(0, 1)

    return torch.hstack((wind, N))

def _make_pairs(
    M: torch.Tensor,
    S: torch.Tensor,
    D: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Given the full time series of bulk momentum profiles, sources, and sinks,
    partition them into input and output entries, and scale both arrays by the
    appropriate budget terms.

    Parameters
    ----------
    M
        Bulk momentum profiles for each time step, bin, and quadrant.
    S
        Added momentum for each time step, bin, and quadrant.
    D
        Dissipiated momentum for each time step and quadrant.

    Returns
    -------
    torch.Tensor, torch.Tensor, torch.Tensor
        Input and output bulk momentum partitions and dissipation profiles, all
        normalized by the budget for each sample.

    """

    M_in = M[:-1].flatten(0, 1)
    M_out = (M - S)[1:].flatten(0, 1)
    D = D[1:].flatten(0, 1)

    budget = M_in.flatten(1, 2).sum(dim=1)[:, None, None]
    return M_in / budget, M_out / budget, D / budget[:, 0]
