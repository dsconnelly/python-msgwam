from os import listdir
from typing import Literal, Iterator, Optional

import numpy as np
import torch
import xarray as xr

from msgwam import config

from ... import hyperparameters as hp
from ...shared.constants import MIMA_MONTHS

from ..architectures import BulkNet

from .transforms import (
    Transform,
    apply_smoothing,
    make_transform,
    reshape_data
)

CMYW = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]

def get_split(
    C: np.ndarray,
    eval_type: Literal['va', 'te'],
    n_samples: Optional[int]=None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get index arrays that can be used to split the data into subsets for
    training and evaluation.

    Parameters
    ----------
    C
        Array of column information as returned by `parse_integrations`. The
        first column of `C` contains a flag which is used to determine whether
        samples come from the training, validation, or held-out test sets, and
        to split the data accordingly.
    eval_type
        Whether the evaluation data should be validation or test data.
    n_samples
        How many samples to return. If `None`, all samples are returned.

    Returns
    -------
    np.ndarray, np.ndarray
        Indices for training and evaluation sets, respectively.

    """

    flag = 1 + (eval_type == 'te')
    idx_tr, = np.where(C[:, 0] < flag)
    idx_ev, = np.where(C[:, 0] == flag)

    if n_samples is not None:
        f = len(idx_tr) / (len(idx_tr) + len(idx_ev))
        n_tr = int(f * n_samples)
        n_ev = n_samples - n_tr

        gen = np.random.default_rng(1234)
        idx_tr = idx_tr[np.argsort(gen.random(len(idx_tr)))[:n_tr]]
        idx_ev = idx_ev[np.argsort(gen.random(len(idx_ev)))[:n_ev]]

    return idx_tr, idx_ev

def iter_paths() -> Iterator[tuple[str, int]]:
    """
    Iterate over all paths to netCDF files that should be read for the given
    source of evaluation data. The strategy is to hold out the months of data
    that we evaluate strategies on, as well as the months just before and after.
    Those data are included only if `eval_type == 'te'`.

    Yields
    ------
    str
        Path to a netCDF file to read.
    int
        A flag that is 0, 1, or 2 depending on whether the path points to an
        integration that is from the training, validation, or test set.

    """

    dt_output = hp.generation.dt_output
    base = f'data/ml-accel/integrations/{dt_output}'
    n_years = len(listdir(base))

    n_va = 1 + (n_years > 1)
    y_min = 25 - n_years + 1
    modulus = n_years * 12

    for site, month_te in MIMA_MONTHS.items():
        month_te = month_te + 12 * (n_years - 1)

        for k, year in enumerate(range(y_min, 26)):
            for m in range(1, 13):
                month = m + k * 12  
                d = month - month_te
                d = min(d % modulus, -d % modulus)

                flag = 2 if d < 2 else (1 if d < 2 + n_va else 0)
                yield f'{base}/{year}/{site}-{m}.nc', flag

def parse_integrations(
    cached: bool=False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load input and target data from the MS-GWaM integrations saved to disk.

    Parameters
    ----------
    cached
        Whether to read the already-concatenated data from disk instead of
        opening the netCDF files. This function must have been called previously
        with `cached` set to `False`, and `path` must be `'va'` or `'te'`.

    Returns
    -------
    ndarray, ndarray, ndarray, ndarray
        Concatenated training and evaluation inputs and targets. The arrays are
        `C` (column information including the mean wind, buoyancy frequency, and
        latitude); `M` (the bulk momentum profile in each phase speed bin); and
        `Y` (the next momentum profile in each phase speed bin along with the
        sink profile). Reshaping and transforming is deferred to trial time,
        except for smoothing, since the Shapiro filter is linear.
    
    """

    base = f'data/ml-accel/cached/{hp.generation.dt_output}'
    make_path = lambda c: f'{base}/{c}.npy'

    if cached:
        return tuple(map(np.load, map(make_path, 'CMY')))

    n_paths = 0
    for _ in iter_paths():
        n_paths = n_paths + 1

    Cs, Ms, Ys = None, None, None
    for i, (path, flag) in enumerate(iter_paths()):
        print(path, flag)
        with xr.open_dataset(path) as ds:
            M, Y, keep = _parse_momentum(ds)
            col = flag * np.ones((M.shape[0], 1))
            C = np.hstack((col, _parse_column(ds)))

            if Cs is None:
                Cs = np.nan * np.zeros((n_paths, *C.shape))
                Ms = np.nan * np.zeros((n_paths, *M.shape))
                Ys = np.nan * np.zeros((n_paths, *Y.shape))

            n_valid = keep.sum()
            Cs[i, :n_valid] = C[keep]
            Ms[i, :n_valid] = M[keep]
            Ys[i, :n_valid] = Y[keep]

    flatten = lambda a: a.reshape(a.shape[0] * a.shape[1], *a.shape[2:])
    Cs, Ms, Ys = flatten(Cs), flatten(Ms), flatten(Ys)    
    keep = ~np.isnan(Cs[:, 0])

    Cs = Cs[keep]
    Ms = Ms[keep]
    Ys = Ys[keep]

    for data, name in zip([Cs, Ms, Ys], 'CMY'):
        np.save(make_path(name), data)

    return Cs, Ms, Ys

def prepare_data(
    n_bins: int,
    eval_type: Literal['va', 'te'],
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
    n_samples: Optional[int]=None,
    transform_inputs: bool=True
) -> tuple[
    CMYW,
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
        Whether the evaluation data is validation or test. Used here to
        generate training and evaluation index arrays.
    arrays
        Tuple of arrays as returned by `parse_integrations`.
    n_samples
        How many samples to return. By default, returns everything.
    transform_inputs
        Whether to actually apply the transforms to the inputs or just return
        them. Defaults to applying them, but can be skipped in plotting.

    Returns
    --------
    ndarray, ndarray, ndarray, ndarray
        Reshaped, filtered, and transformed `C`, `M`, `Y`, and `W` arrays. The
        first column of `C`, containing flags indicating the provenance of each
        sample, will be discarded.
    ndarray, ndarray
        Index arrays splitting the data into training and evaluation sets.
    Transform, Transform
        Transforms for the input arrays. Note that these transforms have already
        been applied to the returned `C` and `M` arrays.
    
    """

    C, M, Y = arrays
    idx_tr, idx_ev = get_split(C, eval_type, n_samples)
    keep = np.concatenate((idx_tr, idx_ev))
    n_tr, n_ev = len(idx_tr), len(idx_ev)
    
    idx = np.arange(n_tr + n_ev)
    C, M, Y = C[keep, 1:], M[keep], Y[keep]
    idx_tr, idx_ev = idx[:n_tr], idx[n_tr:]

    Y, D = Y[:, :-1], Y[:, -1:]
    M, Y = reshape_data(n_bins, M, Y)
    Y = np.concatenate((Y, D), axis=1)

    residual = abs(Y.sum(axis=(1, 2)) - 1).max()
    print(f'Loaded {n_tr} training and {n_ev} evaluation samples.')
    print(f'Maximum residual is {residual:.4e}.')

    C_trans = make_transform(C[idx_tr], mode='z')
    M_trans = make_transform(M[idx_tr], mode=hp.training.M_transform)

    if transform_inputs:
        C = C_trans(C)
        M = M_trans(M)

    W = Y.sum(axis=-1, keepdims=True)
    keep = (W > 0)[..., 0]
    Y[keep] /= W[keep]

    return (C, M, Y, W), (idx_tr, idx_ev), (C_trans, M_trans)

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

    C, M, *_ = parse_integrations(cached=True)
    C, M = torch.as_tensor(C[:10, 1:]), torch.as_tensor(M[:10])

    def trace_func(
        C: torch.Tensor,
        M: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Execute pipeline, excluding momentum budgeting (which can be done at
        integration time) but including input transformations.
        """

        M, = reshape_data(model._n_bins, M)
        Y, W = model(C_trans(C), M_trans(M))
        W = torch.softmax(W, dim=1)
    
        totals = Y.sum(dim=2, keepdim=True)
        totals[totals == 0] = 1
        Y = W * (Y / totals)

        return Y[:, :-1], Y[:, -1]

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
        Arrays of current bulk momentum profiles and still-dimensional momentum
        and sink profiles at the next time step. The last array is an index
        indicating which samples should be retained, so that the corresponding
        rows of the `C` array can be indexed similarly.

    """

    M = ds['M_bulk'].values
    S = ds['source'].values
    D = ds['sink'].values

    Y = (M - S)[1:].reshape(-1, M.shape[2], M.shape[3])
    M = M[:-1].reshape(-1, M.shape[2], M.shape[3])
    D = D[1:].reshape(-1, 1, D.shape[2])
    Y = np.concatenate((Y, D), axis=1)

    budget = M.sum(axis=(1, 2))
    keep, budget = budget > 0, budget[budget > 0, None, None]
    M[keep], Y[keep] = M[keep] / budget, Y[keep] / budget

    for _ in range(hp.training.n_smoothing):
        M = apply_smoothing(M)
        Y = apply_smoothing(Y)

    sink_frac = Y[:, -1].sum(axis=-1)
    residual = abs(1 - Y.sum(axis=(1, 2)))
    keep = keep & (sink_frac < hp.architectures.max_sink)
    keep = keep & (residual < 1e-14)

    return M, Y, keep