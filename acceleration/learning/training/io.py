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

def get_best_task_id() -> int:
    """
    Get the index of the best hyperparameter setting found in grid search.

    Returns
    -------
    int
        Index of the best set of hyperparameters.

    """

    best_loss = np.inf
    best_id = None

    for i in range(hp.grid_size):
        try:
            with open(f'data/ml-accel/records/loss-{i}.txt') as f:
                loss = float(f.read().strip())

            if loss < best_loss:
                best_loss = loss
                best_id = i

        except FileNotFoundError:
            warn(f'Could not find record for hyperparameter setting {i}')

    return best_id

def get_loaders(
    wind: torch.Tensor,
    M: torch.Tensor,
    Y: torch.Tensor,
    idx_tr: torch.Tensor,
    idx_ev: torch.Tensor
) -> tuple[DataLoader, DataLoader]:
    """
    Package the training and evaluation data into `DataLoader` instances.

    Parameters
    ----------
    wind, M, Y
        Tensors of input and output data.
    idx_tr, idx_ev
        Tensors that partition the data into training and evaluation sets.

    Returns
    -------
    DataLoader, DataLoader
        Loaders for the training and evaluation sets.
    
    """

    loaders = []
    for i, idx in enumerate([idx_tr, idx_ev]):
        batch_size = [hp.training.batch_size, 2048][i]
        dataset = TensorDataset(wind[idx], M[idx], Y[idx])
        loaders.append(DataLoader(dataset, batch_size, shuffle=(i == 0)))

    return tuple(loaders)

def get_split(
    n_samples: int,
    eval_type: Literal['va', 'te']
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

    Returns
    -------
    torch.Tensor, torch.Tensor
        Indices for training and evaluation sets, respectively.

    """

    if eval_type == 'va':
        c = int(0.8 * n_samples)
        idx = torch.randperm(n_samples)

    elif eval_type == 'te':
        total = len(_SITES_TR + _SITES_TE)
        c = (n_samples * len(_SITES_TR)) // total
        idx = torch.arange(n_samples)

    else:
        raise ValueError(f'Invalid eval_type: {eval_type}')

    return idx[:c], idx[c:]

def load_tensors(
    eval_type: Literal['va', 'te'],
    n_bins: Optional[int]=None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load input and target data from netCDF files saved to disk.

    Parameters
    ----------
    eval_type
        String specifying whether evaluation data should be from the validation
        set or the test set, used here to determine which files to read.
    n_bins
        How many phase speed bins to preserve. Defaults to the value specified
        by the loaded hyperparameter grid, but can be overridden.

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

    if n_bins is None:
        n_bins = hp.architectures.n_bins

    args = [[], [], []]
    for site in _SITES_TR + _SITES_TE * (eval_type == 'te'):
        with xr.open_dataset(f'data/ml-accel/training/mima-{site}.nc') as ds:
            u = torch.as_tensor(ds['u'].values)
            v = torch.as_tensor(ds['v'].values)
            N = torch.as_tensor(ds['N'].values)

            if len(ds['bin']) % n_bins:
                raise ValueError('Nonconforming bin number:', n_bins)

            div_by = len(ds['bin']) // n_bins
            M = ds['M_bulk'].groupby(ds['bin'] // div_by).sum('bin')
            S = ds['source'].groupby(ds['bin'] // div_by).sum('bin')

            M = torch.as_tensor(M.values).flatten(2, 3)
            S = torch.as_tensor(S.values).flatten(2, 3)
            D = torch.as_tensor(ds['sink'].values)

            for _ in range(hp.training.n_smoothing):
                M = apply_smoothing(M, dim=-1)
                S = apply_smoothing(S, dim=-1)
                D = apply_smoothing(D, dim=-1)

        windN = _make_windN(u, v, N)
        M, Y = _make_pairs(M, S, D)

        args[0].append(windN)
        args[1].append(M)
        args[2].append(Y)

    return tuple(torch.vstack(arg) for arg in args)

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
) -> tuple[torch.Tensor, torch.Tensor]:
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
    torch.Tensor, torch.Tensor
        Input and output bulk momentum partitions normalized by the budget.

    """

    M_in = M[:-1].flatten(0, 1)
    dM = M[1:].flatten(0, 1) - M_in - S[1:].flatten(0, 1)
    Y = torch.hstack((dM, D[1:].flatten(0, 1)))
    budget = M_in.sum(dim=1)[:, None]

    return M_in / budget, Y / budget
