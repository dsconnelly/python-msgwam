from typing import Literal, Iterator, Optional

import numpy as np
import torch
import xarray as xr

from msgwam import config

from ... import hyperparameters as hp

from ..architectures import BulkNet

from .transforms import (
    Transform,
    apply_smoothing,
    make_transform,
    reshape_data
)

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

CMYD = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]

def get_split(
    n_samples: int,
    eval_type: Literal['va', 'te'],
    seed: int=1234
) -> tuple[np.ndarray, np.ndarray]:
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
    np.ndarray, np.ndarray
        Indices for training and evaluation sets, respectively.

    """

    if eval_type == 'va':
        c = int(0.8 * n_samples)
        gen = np.random.default_rng(seed)
        idx = np.argsort(gen.random(n_samples))

    elif eval_type == 'te':
        total = len(_SITES_TR + _SITES_TE)
        c = (n_samples * len(_SITES_TR)) // total
        idx = np.arange(n_samples)

    else:
        raise ValueError(f'Invalid eval_type: {eval_type}')

    return idx[:c], idx[c:]

def iter_paths(eval_type: Literal['va', 'te']) -> Iterator[str]:
    """
    Iterate over all paths to netCDF files that should be read for the given
    source of evaluation data.

    Parameters
    ----------
    eval_type
        If `'te'`, then the held-out locations will be included in the paths to
        read. Otherwise, only the training sites will be read.

    Yields
    ------
    str
        Path to a netCDF file to read.

    """

    base = 'data/ml-accel/integrations'
    sites = _SITES_TR + _SITES_TE * (eval_type == 'te')

    for site in sites:
        for month in range(1, 13):
            yield f'{base}/{site}-{month}.nc'

def parse_integrations(
    eval_type: Literal['va', 'te'],
    cached: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load input and target data from the MS-GWaM integrations saved to disk.

    Parameters
    ----------
    fname
        Whether the evaluation set should be validation or test data. Here used
        to determine which integrations to read.
    cached
        Whether to read the already-concatenated data from disk instead of
        opening the netCDF files. This function must have been called previously
        with `cached` set to `False`, and `path` must be `'va'` or `'te'`.

    Returns
    -------
    ndarray, ndarray, ndarray, ndarray
        Concatenated training and evaluation inputs and targets. The arrays are
        `C` (column information including the mean wind, buoyancy frequency, and
        latitude); `M` (the bulk momentum profile in each phase speed bin); `Y`
        (either the next bulk momentum profile or the change relative to `M`);
        and `D` (the profile of sinks).
    
    """

    base = 'data/ml-accel/cached'
    make_path = lambda c: f'{base}/{c}-{eval_type}.npy'

    if cached:
        return tuple(map(np.load, map(make_path, 'CMYD')))

    stacks = [[], [], [], []]
    for path in iter_paths(eval_type):
        with xr.open_dataset(path) as ds:
            datas = [_parse_column(ds), *_parse_momentum(ds)]
            for stack, data in zip(stacks, datas):
                stack.append(data)

    make_stack = lambda s: np.concatenate(s, axis=0)
    outputs = tuple(map(make_stack, stacks))
    for data, name in zip(outputs, 'CMYD'):
        np.save(make_path(name), data)

    return outputs

def prepare_data(
    n_bins: int,
    eval_type: Literal['va', 'te'],
    arrays: Optional[CMYD]=None
) -> tuple[
    CMYD,
    tuple[np.ndarray, np.ndarray],
    tuple[Transform, Transform]
]:
    """
    Prepare data for training or plotting.

    Parameters
    ----------
    n_bins
        Number of bins that should be in the transformed data.
    eval_type
        Whether the evaluation set is validation or test data. Needed here to
        determine how to split the datasets.
    arrays
        Tuple of arrays as returned by `parse_integrations`. If `CMYD` is not
        provided, `parse_integrations` is called here.

    Returns
    --------
    ndarray, ndarray, ndarray, ndarray
        Reshaped, filtered, and transformed `C`, `M`, `Y`, and `D` arrays.
    ndarray, ndarray
        Index arrays splitting the data into training and evaluation sets.
    Transform, Transform
        Transforms for the input arrays. Note that these transforms have already
        been applied to the returned `C` and `M` arrays.
    
    """

    if arrays is None:
        arrays = parse_integrations(eval_type)

    C, M, Y, D = arrays
    M, Y = reshape_data(n_bins, M, Y)
    idx_tr, idx_ev = get_split(M.shape[0], eval_type)

    for _ in range(hp.training.n_smoothing):
        M = apply_smoothing(M)
        Y = apply_smoothing(Y)
        D = apply_smoothing(D)

    n_tr, n_ev = len(idx_tr), len(idx_ev)
    print(f'Loaded {n_tr} training and {n_ev} evaluation samples.')

    budget = Y.sum(axis=(1, 2)) + D.sum(1)
    residual = abs(budget - (not hp.architectures.learn_delta))
    print(f'Maximum residual is {residual.max():.4e}.')

    C_trans = make_transform(C[idx_tr], mode='z')
    M_trans = make_transform(M[idx_tr], mode=hp.training.M_transform)
    C, M = C_trans(C), M_trans(M)

    return (C, M, Y, D), (idx_tr, idx_ev), (C_trans, M_trans)

def trace(
    model: BulkNet,
    C_trans: Transform,
    M_trans: Transform
) -> torch.jit.ScriptFunction:
    """
    Trace the model pipeline and return a JITted object that can be evaluated in
    MS-GWaM without having the `BulkNet` class definition available.

    Parameters
    ----------
    model
        Trained `BulkNet` instance.
    C_trans, M_trans
        Transforms used during training on the model inputs.

    Returns
    -------
    ScriptFunction
        Traced model pipeline.

    """

    cpu = torch.device('cpu')
    model.eval().to(cpu)

    for p in model.parameters():
        p.requires_grad = False

    C, M, *_ = parse_integrations('va')
    C = torch.as_tensor(C[:10])
    M = torch.as_tensor(M[:10])

    def trace_func(
        C: torch.Tensor,
        M: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Execute pipeline, excluding momentum budgeting (which can be done at
        integration time) but including input transformations.
        """

        M, = reshape_data(model._n_bins, M)
        return model(C_trans(C), M_trans(M))
    
    with torch.no_grad():
        return torch.jit.trace(trace_func, (C, M))

def _parse_column(ds: xr.Dataset) -> np.ndarray:
    """
    Parse a `Dataset` to build a context array.

    Parameters
    ----------
    ds
        Loaded dataset containing integration outputs.

    Returns
    -------
    np.ndarray
        Array whose first dimension ranges over samples and whose second ranges
        first over velocity, than buoyancy frequency, and then the latitude.

    """

    u = ds['u'].values[:, None]
    v = ds['v'].values[:, None]
    N = ds['N'].values[:, None]

    quad = np.arange(4)[None, :, None]
    u, v, N, quad = np.broadcast_arrays(u, v, N, quad)

    is_zonal = (np.remainder(quad, 2) == 0)
    wind = is_zonal * u + (1 - is_zonal) * v
    wind[quad > 1] = -wind[quad > 1]

    shape = (-1, config.n_grid - 1)
    wind, N = wind[:-1].reshape(*shape), N[:-1].reshape(*shape)
    lat = ds.attrs['latitude'] * np.ones((wind.shape[0], 1))

    return np.hstack((wind, N, lat))

def _parse_momentum(
    ds: xr.Dataset
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse a `Dataset` for relevant information about the bulk momentum.

    Parameters
    ----------
    ds
        Loaded dataset containing integration outputs.

    Returns
    -------
    ndarray, ndarray, ndarray
        Arrays of current bulk momentum profiles, bulk momentum profiles or
        deltas at the next times step, and dissipation profiles.

    """

    M = ds['M_bulk'].values
    S = ds['source'].values
    D = ds['sink'].values

    Y = (M - S)[1:].reshape(-1, M.shape[2], M.shape[3])
    M = M[:-1].reshape(-1, M.shape[2], M.shape[3])
    D = D[1:].reshape(-1, D.shape[2])

    if hp.architectures.learn_delta:
        Y = Y - M

    budget = M.sum(axis=(1, 2))
    keep, budget = budget > 0, budget[budget > 0, None, None]
    return M[keep] / budget, Y[keep] / budget, D[keep] / budget[:, 0]
