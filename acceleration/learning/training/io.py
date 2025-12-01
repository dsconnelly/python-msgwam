from os import listdir
from typing import Any, Literal, Iterator, Optional

import numpy as np
import torch
import xarray as xr

from msgwam import config

from optuna import load_study
from optuna.trial import FixedTrial, Trial

from ... import hyperparameters as hp
from ...shared.constants import MIMA_MONTHS

from ..generation import get_bin_edges

from .reconstruction import correct_bins, get_dM, get_vertical_flux
from .transforms import (
    Transform,
    apply_smoothing,
    reshape_data,
)

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

    base = f'data/ml-accel/integrations-{hp.generation.dt_output}'
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

def cache_arrays(n_bins_str: str) -> None:
    """
    Load input and target data from the MS-GWaM integrations saved to disk.

    Parameters
    ----------
    n_bins
        How many bins include in the returned data.

    """

    n_bins = int(n_bins_str)
    n_smoothing = hp.training.n_smoothing
    make_path = lambda c: f'data/ml-accel/cached/{n_smoothing}/{c}-{n_bins}.npy'

    n_paths = 0
    for _ in iter_paths():
        n_paths = n_paths + 1

    Cs, Ms, Ys = None, None, None
    for i, (path, flag) in enumerate(iter_paths()):
        with xr.open_dataset(path) as ds:
            C = _parse_column(ds)
            u_old, u_new = C[:, 0], C[:, -1]
            C = C[:, :-1].reshape(C.shape[0], -1)

            col = flag * np.ones((C.shape[0], 1))
            lat = ds.attrs['latitude'] * np.ones((C.shape[0], 1))
            M, Y, budget, keep = _parse_momentum(ds, n_bins, u_old, u_new)
            C = np.hstack((col, C, lat, budget))

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
    tuple[Transform, Transform, Transform]
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
    n_smoothing = hp.training.n_smoothing

    for name in 'CMY':
        path = f'data/ml-accel/cached/{n_smoothing}/{name}-{n_bins}.npy'
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

    p_M = trial.suggest_int('p_M', 1, 5)
    p_F = trial.suggest_int('p_F', 3, 5)
    p_D = trial.suggest_int('p_D', 3, 5)
    p_Y = torch.as_tensor([p_F, p_D])[:, None, None]

    C, meta = C[:, :-2], C[:, -2:]
    C = C.reshape(-1, 2, config.n_grid - 1)
    
    ones = torch.ones(config.n_grid - 1)
    C_mean = C.mean(dim=(0, 2))[:, None] * ones
    C_std = C.std(dim=(0, 2))[:, None] * ones

    C_mean = torch.cat((C_mean.flatten(), meta.mean(0)))
    C_std = torch.cat((C_std.flatten(), meta.std(0)))
    C = torch.hstack((C.flatten(1, 2), meta))

    C_trans = Transform((C_mean, C_std)).float()
    M_trans = Transform(M[idx_tr], True, True, p_M).float()
    Y_trans = Transform(Y[idx_tr], False, True, p_Y).float()

    if apply_transforms:
        C = C_trans(C)
        M = M_trans(M)

    dM = get_dM(Y)
    Y = torch.cat((Y_trans(Y), dM[:, None]), dim=1)

    return (C, M, Y), (idx_tr, idx_ev), (C_trans, M_trans, Y_trans)

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
    u_old = wind[:-1].reshape(*shape)
    u_new = wind[1:].reshape(*shape)
    N = N[:-1].reshape(*shape)

    return np.stack((u_old, N, u_new), axis=1)

def _parse_momentum(
    ds: xr.Dataset,
    n_bins: int,
    u_old: np.ndarray,
    u_new: np.ndarray,
    mode: str='from_left'
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse the bulk momentum density variables on a dataset and return the
    relevant training inputs and targets.
    """

    M = ds['M_bulk'].values
    S = ds['source'].values
    D = ds['sink'].values

    M_in = M[:-1].reshape(-1, *M.shape[2:])
    M_out = (M - S)[1:].reshape(-1, *M.shape[2:])
    D = -D[1:].reshape(-1, *D.shape[2:])

    M_in = reshape_data(M_in, n_bins, mode)
    M_out = reshape_data(M_out, n_bins, mode)
    D = reshape_data(D, n_bins, mode)

    if hp.training.correct_bins:
        edges = get_bin_edges(n_bins, mode)
        M_out = correct_bins(M_out, u_new, u_old, edges)
        D = correct_bins(D, u_new, u_old, edges)

    F_est = ds['F_bulk'].values
    F_est = F_est[1:].reshape(-1, *F_est.shape[2:])
    F_est = reshape_data(F_est, n_bins, 'from_left')

    M_tot = reshape_data(M[:-1], n_bins, mode).sum(axis=1)[:, None]
    M_tot = np.broadcast_to(M_tot, (M_tot.shape[0], 4, *M_tot.shape[2:]))
    M_tot = M_tot.reshape(-1, *M_tot.shape[2:]) - M_in

    for _ in range(hp.training.n_smoothing):
        M_in = apply_smoothing(M_in)
        M_out = apply_smoothing(M_out)
        M_tot = apply_smoothing(M_tot)
        
        F_est = apply_smoothing(F_est)
        D = apply_smoothing(D)

    dF = (M_out - M_in - D).sum(1)
    F = get_vertical_flux(dF, F_est)
    Y = np.stack((F[..., 1:], D), axis=1)

    budget = M_in.sum(axis=(1, 2))
    M_in = np.concatenate((M_in, M_tot), axis=1)
    keep = budget > 0

    sink = Y[:, 0, :, -1].sum(-1) - Y[:, 1].sum((1, 2))
    res = M_out.sum((1, 2)) + sink - budget
    keep = keep & (abs(res) < 1e-14)

    n_active = (abs(Y.sum(axis=(-2, -1))) > 1e-12).sum(-1)
    keep = keep & (n_active == 2)

    budget[keep] = np.log(budget[keep])
    return M_in, Y, budget[:, None], keep
