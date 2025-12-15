from os import listdir
from typing import Any, Literal, Iterator, Optional

import numpy as np
import torch
import xarray as xr

from msgwam import config
from msgwam.constants import ROT_EARTH

from optuna import load_study
from optuna.trial import FixedTrial, Trial

from ... import hyperparameters as hp
from ...shared.constants import MIMA_MONTHS
from ...shared.distributed import add_task_info, combine, get_workload

from .reconstruction import invert_cg
from .transforms import Transform, apply_smoothing, get_T_from_logits

CMY = tuple[torch.Tensor, torch.Tensor, torch.Tensor]

def get_split(
    flag: np.ndarray,
    eval_type: Literal['va', 'te'],
    n_samples: Optional[int]=None,
    seed: int=1234
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
    seed
        Seed to use if retaining fewer than all samples.

    Returns
    -------
    np.ndarray, np.ndarray
        Indices for training and evaluation sets, respectively.

    """

    target = 1 + (eval_type == 'te')
    idx_tr, = np.where(flag < target)
    idx_ev, = np.where(flag == target)

    if n_samples is not None:
        f = len(idx_tr) / (len(idx_tr) + len(idx_ev))
        n_tr = int(f * n_samples)
        n_ev = n_samples - n_tr

        gen = np.random.default_rng(seed)
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

    base = f'data/ml-accel/integrations-cg'
    n_years = len(listdir(base))

    n_va = 1 + (n_years > 1)
    y_min = 25 - n_years + 1
    modulus = n_years * 12

    n_tasks = len(MIMA_MONTHS) * n_years * 12
    start, end = get_workload(n_tasks)
    i = -1

    for site, month_te in MIMA_MONTHS.items():
        month_te = month_te + 12 * (n_years - 1)

        for k, year in enumerate(range(y_min, 26)):
            for m in range(1, 13):
                i = i + 1
                if not (start <= i < end):
                    continue

                month = m + k * 12  
                d = month - month_te
                d = min(d % modulus, -d % modulus)

                flag = 2 if d < 2 else (1 if d < 2 + n_va else 0)
                yield f'{base}/{year}/{site}-{m}.nc', flag

def cache_arrays(n_bins_str: str, mode: Literal['compute', 'combine']) -> None:
    """
    Load input and target data from the MS-GWaM integrations saved to disk.

    Parameters
    ----------
    n_bins
        How many bins include in the returned data.

    """

    n_bins = int(n_bins_str)
    n = hp.training.n_smoothing

    base = f'data/ml-accel/cached-cg/{n}'
    make_path = lambda c: add_task_info(f'{base}/{c}-{n_bins}.npy')

    if mode == 'combine':
        combine(make_path('C'), remove_after=True)
        combine(make_path('M'), remove_after=True)
        combine(make_path('Y'), remove_after=True)

        return

    n_paths = 0
    for _ in iter_paths():
        n_paths = n_paths + 1

    Cs, Ms, Ys = None, None, None
    for i, (path, flag) in enumerate(iter_paths()):
        with xr.open_dataset(path) as ds:
            C = _parse_column(ds)
            ones = np.ones((C.shape[0], 1))
            f = 2 * ROT_EARTH * np.sin(ds.attrs['latitude'])
            C = np.hstack((flag * ones, C, abs(f) * ones))

            N, f = C[:, -config.n_grid:-1], C[:, -1]
            M, Y, keep = _parse_momentum(ds, N, f)

            print(f'{path}: found {keep.sum()} samples')

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

    print(f'Cached {keep.sum()} total samples')

def get_best_trial(
    exp_name: Optional[str]=None,
    **kwargs: dict[str, Any]
) -> FixedTrial:
    """
    Load a `Trial` object with the best hyperparameters found during a sweep
    from a study saved to database storage.

    Parameters
    ----------
    exp_name
        Name of the experiment to load. If `None`, defaults to the currently
        loaded value from the hyperparameter file.
    kwargs
        Parameters to override. If none are provided, then the exact set of
        hyperparameters found during the sweep is returned.

    Returns
    -------
    FixedTrial
        Wrapper around the hyperparameters, for use in training or plotting.

    """

    exp_name = hp.training.exp_name if exp_name is None else exp_name
    path = f'sqlite:///data/ml-accel/models/study-{exp_name}.db'
    
    try:
        study = load_study(study_name=f'{exp_name}-large', storage=path)

    except KeyError:
        study = load_study(study_name=f'{exp_name}-small', storage=path)
        print('Found only a warmup study; using that instead.')

    params = study.best_params
    params.update(**kwargs)

    return FixedTrial(params)

def prepare_data(
    trial: Trial,
    n_bins: int,
    eval_type: Literal['va', 'te'],
    n_samples: Optional[int]=None,
    apply_transforms: bool=True,
    seed: int=1234
) -> tuple[
    CMY,
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
    n_samples
        How many samples to return. By default, returns everything.
    apply_transforms
        Whether to actually apply the transforms to the tensors or just return
        them. Defaults to applying them, but can be skipped in plotting.
    seed
        Seed to use if subsetting from the available data.

    Returns
    --------
    Tensor, Tensor, Tensor, Tensor
        Reshaped, filtered, and transformed `C`, `M`, `Y`, and `W` arrays. The
        first column of `C`, containing flags indicating the provenance of each
        sample, will be discarded.
    ndarray, ndarray
        Index arrays splitting the data into training and evaluation sets.
    Transform, Transform
        Transforms for the input arrays. Note that these transforms may have
        already been applied to the returned `C` and `M` arrays.
    
    """

    memmaps = []
    n = hp.training.n_smoothing

    for name in 'CMY':
        path = f'data/ml-accel/cached-cg/{n}/{name}-{n_bins}.npy'
        memmaps = memmaps + [np.load(path, mmap_mode='r')]

    C_mm, M_mm, Y_mm = memmaps
    idx_tr, idx_ev = get_split(C_mm[:, 0], eval_type, n_samples, seed)
    keep = np.concatenate((idx_tr, idx_ev))
    n_tr, n_ev = len(idx_tr), len(idx_ev)

    sdx = np.argsort(keep)
    idx_tr, = np.where(sdx < n_tr)
    idx_ev, = np.where(sdx >= n_tr)
    keep = keep[sdx]

    C = torch.as_tensor(C_mm[keep, 1:]).float()
    M = torch.as_tensor(M_mm[keep]).float()
    Y = torch.as_tensor(Y_mm[keep]).float()

    print(f'Found {n_tr} training and {n_ev} evaluation samples.')
    del C_mm, M_mm, Y_mm

    C, meta = C[:, :-1], C[:, -1:]
    C = C.reshape(-1, 2, config.n_grid - 1)

    ones = torch.ones(config.n_grid - 1)
    C_mean = C.mean(dim=(0, 2))[:, None] * ones
    C_std = C.std(dim=(0, 2))[:, None] * ones

    C_mean = torch.cat((C_mean.flatten(), meta.mean(0)))
    C_std = torch.cat((C_std.flatten(), meta.std(0)))
    C = torch.hstack((C.flatten(1, 2), meta))

    p_M = trial.suggest_int('p_M', 1, 5)
    C_trans = Transform((C_mean, C_std)).float()
    M_trans = Transform(M[idx_tr], True, True, p_M).float()

    N = C[:, None, -config.n_grid:-1]
    f = C[:, -1, None, None]

    logits = get_T_from_logits(N, f, Y[:, 0], inverse=True)
    Y = torch.stack((logits, Y[:, 1]), dim=1)

    if apply_transforms:
        C = C_trans(C)
        M = M_trans(M)

    return (N, f, C, M, Y), (idx_tr, idx_ev), (C_trans, M_trans)

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
    wind = wind[:-1].reshape(*shape)
    N = N[:-1].reshape(*shape)

    return np.hstack((wind, N))

def _parse_momentum(
    ds: xr.Dataset,
    N: np.ndarray,
    f: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse the bulk momentum density variables on a dataset and return the
    relevant training inputs and targets.
    """

    M = ds['M_bulk'].values[:-1]
    cg = ds['cg_bulk'].values[:-1]
    shape = (M.shape[0] * M.shape[1], *M.shape[2:])
    M, cg = M.reshape(*shape), cg.reshape(*shape)

    cg = (M * cg).sum(axis=1)
    M = M.sum(axis=1)

    idx = M > 0
    cg[idx] = cg[idx] / M[idx]

    for _ in range(hp.training.n_smoothing):
        M = apply_smoothing(M)
        cg = apply_smoothing(cg)

    T_hat, keep = invert_cg(N, f, cg)
    Y = np.stack((T_hat, cg), axis=1)

    return M, Y, keep
