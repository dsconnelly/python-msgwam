from __future__ import annotations
from typing import TYPE_CHECKING

import torch
import xarray as xr

from torch.nn.functional import pad
from tqdm import tqdm, trange

from . import config

if TYPE_CHECKING:
    from .rays import RayCollection

def get_fracs(
    rays: RayCollection,
    edges: torch.Tensor
) -> torch.Tensor:
    """
    Given the properties of the ray volumes (used to find the trailing and
    leading edges of each volume) along with the edges of the vertical grid
    cells, find the fraction of each grid cell that is intersected by each ray.

    Parameters
    ----------
    rays
        Collection of current ray volume properties.
    edges
        Edges of the vertical grid cells.

    Returns
    -------
    torch.Tensor
        Tensor of shape (len(edges) - 1, config.n_ray_max), where the value at
        index [i, j] corresponds to the fraction of cell i that is intersected
        by ray volume j.
    
    """

    r_lo = rays.r - 0.5 * rays.dr
    r_hi = rays.r + 0.5 * rays.dr
    r_mins = torch.maximum(r_lo, edges[:-1, None])
    r_maxs = torch.minimum(r_hi, edges[1:, None])

    return torch.clamp(r_maxs - r_mins, min=0) / (edges[1] - edges[0])

def get_iterator() -> range | tqdm[int]:
    """
    If `config.show_progress`, returns a tqdm object configured to show useful
    output in the terminal. Otherwise, just return an appropriate `range`.

    Returns
    -------
    range | tqdm[int]
        An iterator from 1 to `config.n_t_max`.

    """

    if not config.show_progress:
        return range(1, config.n_t_max)
    
    format = (
        '{desc}: {percentage:3.0f}%|' +
        '{bar}' +
        '| {n:.2f}/{total_fmt} [{rate_fmt}{postfix}]'
    )

    return trange(
        1, config.n_t_max,
        bar_format=format,
        unit_scale=(config.dt / 86400),
        unit='day'
    )

def interp(
    x_new: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor
) -> torch.Tensor:
    """
    One-dimensional piecewise linear interpolation. Given a dataset {(x, y)},
    return the values of the piecewise-linear interpolant evaluated at the new
    coordinates contained in `x_new`. This function is intended to function in
    the same way as `numpy.interp`, but allow gradients to propagate through.

    Parameters
    ----------
    x_new
        Coordinates at which to evaluate the interpolant.
    x
        x coordinates of the points in the known data. Must be sorted.
    y
        y coordinates of the points in the known data.

    Returns
    -------
    torch.Tensor
        Interpolated values. Note that, unlike `numpy.interp`, this function
        extrapolates beyond the given dataset assuming constant slope.
        
    """

    dx = x[1:] - x[:-1]
    diff = (x_new[:, None] - x)[:, :-1] / dx
    mask = torch.clamp(torch.inf * -diff, min=0)

    idx = torch.arange(len(x_new))
    jdx = torch.argmin(diff + mask, dim=1)
    factor = diff[idx, jdx]

    return (1 - factor) * y[jdx] + factor * y[jdx + 1]

def open_dataset(*args, **kwargs) -> xr.Dataset:
    """
    Open a netCDF file as an xarray Dataset. This function exists simply so that
    use_cftime=True can be the default.

    Parameters
    ----------
    args
        Positional arguments to xr.open_dataset.
    kwargs
        Keyword arguments to xr.open_dataset.

    Returns
    -------
    xr.Dataset
        The opened Dataset object, opened using cftime.

    """

    return xr.open_dataset(*args, use_cftime=True, **kwargs)

def pad_ends(data: torch.Tensor) -> torch.Tensor:
    """
    Pad both ends of a one-dimensional tensor by repeating the start and end
    values. This function exists because the syntax for torch.nn.functional.pad
    is somewhat complicated for one-dimensional tensors.

    Parameters
    ----------
    data
        Tensor to be padded.

    Returns
    -------
    torch.Tensor
        Padded tensor. Has two more elements than the tensor passed in.

    """

    return pad(data.view(1, -1), (1, 1), mode='replicate').flatten()

def put(a: torch.Tensor, i: int, values: torch.Tensor):
    """
    Wrapper around Tensor.index_put, useful for assigning rows of tensors while
    maintaining differentiability.

    Parameters
    ----------
    a
        Array to be cloned and then rewritten.
    i
        Row to be overwritten.
    values
        New data to write in row `i`.

    """

    return a.index_put((torch.tensor(i),), values)

def shapiro_filter(data: torch.Tensor) -> torch.Tensor:
    """
    Apply a zeroth-order Shapiro filter.

    Parameters
    ----------
    data
        Tensor to be filtered.

    Returns
    -------
    torch.Tensor
        Filtered tensor. Has two fewer elements than the tensor passed in.

    """

    return (data[:-2] + 2 * data[1:-1] + data[2:]) / 4
