from os import listdir
from typing import Literal, Iterator, Optional

import numba as nb
import numpy as np
import torch
import xarray as xr

from msgwam import config

from ... import hyperparameters as hp
from ...shared.constants import MIMA_MONTHS

from .transforms import (
    Transform,
    apply_smoothing,
    reshape_data
)

CMYW = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]

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

    base = f'data/ml-accel/integrations'
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
    make_path = lambda c: f'data/ml-accel/cached/{c}-{n_bins}.npy'

    n_paths = 0
    for _ in iter_paths():
        n_paths = n_paths + 1

    Cs, Ms, Fs = None, None, None
    for i, (path, flag) in enumerate(iter_paths()):
        with xr.open_dataset(path) as ds:
            print(path, flag)

            M, F, budget, keep = _parse_momentum(ds, n_bins)
            col = flag * np.ones((M.shape[0], 1))
            C = np.hstack((col, _parse_column(ds), budget))

            if Cs is None:
                Cs = np.nan * np.zeros((n_paths, *C.shape))
                Ms = np.nan * np.zeros((n_paths, *M.shape))
                Fs = np.nan * np.zeros((n_paths, *F.shape))

            n_valid = keep.sum()
            Cs[i, :n_valid] = C[keep]
            Ms[i, :n_valid] = M[keep]
            Fs[i, :n_valid] = F[keep]

    flatten = lambda a: a.reshape(a.shape[0] * a.shape[1], *a.shape[2:])
    Cs, Ms, Fs = flatten(Cs), flatten(Ms), flatten(Fs)    
    keep = ~np.isnan(Cs[:, 0])

    Cs = Cs[keep]
    Ms = Ms[keep]
    Fs = Fs[keep]

    for data, name in zip([Cs, Ms, Fs], 'CMF'):
        np.save(make_path(name), data)

    print(f'Cached {keep.sum()} total samples')

def prepare_data(
    n_bins: int,
    eval_type: Literal['va', 'te'],
    n_samples: Optional[int]=None,
    transform_inputs: bool=True,
    seed: int=1234
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
    n_samples
        How many samples to return. By default, returns everything.
    transform_inputs
        Whether to actually apply the transforms to the inputs or just return
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
    for name in 'CMF':
        path = f'data/ml-accel/cached/{name}-{n_bins}.npy'
        memmaps = memmaps + [np.load(path, mmap_mode='r')]

    C_mm, M_mm, F_mm = memmaps
    idx_tr, idx_ev = get_split(C_mm[:, 0], eval_type, n_samples, seed)
    keep = np.concatenate((idx_tr, idx_ev))
    n_tr, n_ev = len(idx_tr), len(idx_ev)

    sdx = np.argsort(keep)
    idx_tr, = np.where(sdx < n_tr)
    idx_ev, = np.where(sdx >= n_tr)
    keep = keep[sdx]

    C = torch.as_tensor(C_mm[keep, 1:])
    M = torch.as_tensor(M_mm[keep])
    F = torch.as_tensor(F_mm[keep])

    print(f'Found {n_tr} training and {n_ev} evaluation samples.')
    del C_mm, M_mm, F_mm

    C_trans = Transform(C[idx_tr], mode='z')
    M_trans = Transform(M[idx_tr], mode=hp.training.M_transform)

    if transform_inputs:
        C = C_trans(C)
        M = M_trans(M)

    W = abs(F.sum(dim=-1, keepdim=True))
    tensors = [C, M, F / W, hp.training.W_scale * W]
    tensors = tuple([a.float() for a in tensors])

    return tensors, (idx_tr, idx_ev), (C_trans, M_trans)

@nb.njit
def _fix_vertical_fluxes(
    deltas: np.ndarray,
    F_v: np.ndarray
) -> None:
    """
    The projected group velocities give some idea of the vertical fluxes, but
    they are not always accurate with respect to the observed changes in `M`.
    Here we enforce vertical level-wise balance by scaling the vertical fluxes.

    Parameters
    ----------
    deltas
        Effective change at each grid cell, equal to `dM + D` summed over the
        phase speed bins.
    F_v
        Vertical fluxes from group velocity projection. Will be overwritten.

    """

    for i in range(F_v.shape[0]):
        for k in range(F_v.shape[2]):
            F_b = 0 if k == 0 else F_v[i, :, k - 1].sum()
            F_needed = F_b - deltas[i, k]
            F_out = F_v[i, :, k].sum()

            if abs(F_needed) < 1e-16:
                F_needed = 0

            if F_out != 0:
                F_v[i, :, k] *= F_needed / F_out
                
            elif F_needed != 0:
                F_v[i, :, k] = F_needed / F_v.shape[1]

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
    ds: xr.Dataset,
    n_bins: int
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

    F_v = ds['F_bulk'].values[..., 1:]
    dz = np.diff(ds['z_faces'].values)[0]
    F_v = F_v * hp.generation.dt_output / dz
    F_v = F_v[1:].reshape(-1, *M.shape[2:])

    M_in = M[:-1].reshape(-1, *M.shape[2:])
    M_out = (M - S)[1:].reshape(-1, *M.shape[2:])
    D = D[1:].reshape(-1, *D.shape[2:])
    dM = M_out - M_in

    M_in = reshape_data(M_in, n_bins, 'sum')
    _fix_vertical_fluxes((dM + D).sum(1), F_v)
    F = reshape_data(np.stack((F_v, -D), axis=1), n_bins, 'sum')

    for _ in range(hp.training.n_smoothing):
        M_in = apply_smoothing(M_in)
        F = apply_smoothing(F)

    budget = M_in.sum(axis=(1, 2))
    keep, budget = budget > 0, budget[:, None, None]

    M_in[keep] = M_in[keep] / budget[keep]
    M_out[keep] = M_out[keep] / budget[keep]
    F[keep] = F[keep] / budget[keep, None]
    budget[keep] = np.log(budget[keep])

    res = M_out.sum(axis=(1, 2)) - F[:, -1].sum((1, 2)) - 1
    n_active = (abs(F.sum(axis=-1)) > 1e-8).sum(axis=(-2, -1))
    keep = keep & (abs(res) < 1e-14) & (n_active == 2 * n_bins)
    F[abs(F) < 1e-14] = 0

    return M_in, F, budget[:, 0], keep