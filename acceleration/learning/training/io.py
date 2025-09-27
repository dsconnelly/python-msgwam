from typing import Literal

import numpy as np
import torch
import xarray as xr

from torch.utils.data import DataLoader, TensorDataset

from msgwam.utils import get_vertical_grids

from ...hyperparameters import training as hp

_SITES_TR = [
    # 'anchorage',
    # 'new-york',
    'lisbon',
    # 'miami',
    # 'maldives',
    # 'brisbane',
    # 'buenos-aires',
    # 'weddell-sea'
]

_SITES_TE = [
    'copenhagen',
    'singapore',
    'perth',
    'amundsen-sea'
]

def get_loaders(
    M: torch.Tensor,
    cg: torch.Tensor,
    wind: torch.Tensor,
    targets: torch.Tensor,
    idx_tr: torch.Tensor,
    idx_ev: torch.Tensor
) -> tuple[DataLoader, DataLoader]:
    """
    Package the training and evaluation data into `DataLoader` instances.

    Parameters
    ----------
    

    Returns
    -------
    """

    loaders = []
    for i, idx in enumerate([idx_tr, idx_ev]):
        batch_size = hp.batch_size if i == 0 else 2048
        dataset = TensorDataset(M[idx], cg[idx], wind[idx], targets[idx])
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

    else:
        total = len(_SITES_TR + _SITES_TE)
        c = (n_samples * len(_SITES_TR)) // total
        idx = torch.arange(n_samples)

    return idx[:c], idx[c:]

def load_tensors(
    eval_type: Literal['va', 'te']
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
    
    z, _ = get_vertical_grids()
    dz = np.diff(z)[0] * np.ones_like(z)
    dz = torch.as_tensor(dz)

    args = [[], [], [], []]
    for site in _SITES_TR + _SITES_TE * (eval_type == 'te'):
        with xr.open_dataset(f'data/ml-accel/training/mima-{site}.nc') as ds:
            u = torch.as_tensor(ds['u'].values)
            v = torch.as_tensor(ds['v'].values)
            wind = _get_wind(u, v)

            M = torch.as_tensor(ds['M_bulk'].values)
            F = torch.as_tensor(ds['F_bulk'].values)
            source = torch.as_tensor(ds['source'].values)

        for _ in range(hp.n_smoothing):
            M = _apply_smoothing(M)
            F = _apply_smoothing(F)

        cg = torch.zeros_like(M)
        cg[M > 0] = F[M > 0] / M[M > 0]
        M = M * dz

        wind = _get_wind(u, v)
        M_in, M_out = _get_Ms(M, source)
        cg_in = cg[:-1].flatten(0, 1)
        cg_out = cg[1:].flatten(0, 1)
        
        args[0].append(M_in)
        args[1].append(cg_in)
        args[2].append(wind)

        targets = torch.hstack((M_out, cg_out))
        args[3].append(targets)

    return tuple(torch.vstack(arg) for arg in args)

def _apply_smoothing(a: torch.Tensor) -> torch.Tensor:
    """
    Apply a Shapiro (diffusion) filter along the last dimension of a tensor.

    Parameters
    ----------
    a
        Tensor to smooth.

    Returns
    -------
    torch.Tensor
        Smoothed data.

    """

    out = a.clone()
    out = out.transpose(0, -1)
    left = 3 * out[0] + out[1]
    right = out[-2] + 3 * out[-1]
 
    out[1:-1] = out[:-2] + 2 * out[1:-1] + out[2:]
    out[0], out[-1] = left, right

    return out.transpose(0, -1) / 4

def _get_wind(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Get the appropriate component of the mean wind at each sample, and negate
    the wind profile for samples with negative wavenumber

    Parameters
    ----------
    u, v
        Zonal and meridional components of the mean wind, respectively.

    Returns
    -------
    torch.Tensor
        Wind profile to use in predicting each sample.

    """

    u, v = u[:, None], v[:, None]
    quad = torch.arange(4)[None, :, None]
    u, v, quad = torch.broadcast_tensors(u, v, quad)

    is_zonal = (torch.remainder(quad, 2) == 0).int()
    wind = is_zonal * u + (1 - is_zonal) * v
    wind[quad > 1] = -wind[quad > 1]

    return wind[:-1].flatten(0, 1)

def _get_Ms(
    M: torch.Tensor,
    source: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given the full time series of bulk momentum profiles, partition it into
    input and output profiles, and scale both by the appropriate budget terms
    (including new momentum from the source).

    Parameters
    ----------
    M
        Bulk momentum profiles for each time step and quadrant.
    source
        Added momentum for each time step and quadrant.

    Returns
    -------
    torch.Tensor, torch.Tensor
        Input and output bulk momentum profiles normalized by the budget. The
        input profile has the (normalized) source momentum as its last term.

    """

    M_in = M[:-1].flatten(0, 1)
    M_out = M[1:].flatten(0, 1)

    source = source[:-1].flatten(0, 1)
    M_in = torch.hstack((M_in, source[:, None]))
    budget = M_in.sum(axis=1)[:, None]

    return M_in / budget, M_out / budget
