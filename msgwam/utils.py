from __future__ import annotations
from abc import ABC
from itertools import takewhile
from typing import TYPE_CHECKING, Iterator, Optional, Self, TypeVar

import cftime
import numpy as np
import torch
import xarray as xr

from scipy.ndimage import gaussian_filter1d as _filter
from tqdm import trange

from . import config
from .constants import EPOCH
from .dispersion import get_m

if TYPE_CHECKING:
    from numpy.random import Generator
    from .means import PrescribedWind

    _T = TypeVar('T')

class FactoryABC(ABC):
    """
    Abstract base class that allows the particular subclass to be created to be
    determined by a string in a configuration file.
    """

    @classmethod
    def from_name(cls, name: str, *args, **kwargs) -> Self:
        """
        Return an instance of the subclass of this class indicated by `name`.

        Parameters
        ----------
        name
            Indicates the subclass to instantiate. Subclasses are assumed to be
            named in Pascal case, and matches are sought against the first word
            of each subclass name.
        args, kwargs
            Arguments to be passed to the subclass constructor.

        Returns
        -------
        Self
            Instantiated subclass.

        """

        def get_subs(cls):
            """Recursively find subclasses."""

            out = set()
            for sub in cls.__subclasses__():
                out.update({sub} | get_subs(sub))

            return out

        is_lower = lambda c: c.islower()
        handle = lambda s: s[0].lower() + ''.join(takewhile(is_lower, s[1:]))
        subs = {handle(sub.__name__) : sub for sub in get_subs(cls)}

        return subs[name](*args, **kwargs)
    
def cos_and_sin(a: np.ndarray) -> tuple[np.ndarray]:
    """
    Convenience function for computing both trigonometric functions at the same
    time and rounding them to ten decimal places, such that when evaluated at
    multiples of `np.pi / 2` the results will be exactly zero or one.

    Parameters
    ----------
    a
        Array of values in radians.

    Returns
    -------
    np.ndarray, np.ndarray
        Array of cosine and sine values, respectively.

    """

    cos = np.round(np.cos(a), 10)
    sin = np.round(np.sin(a), 10)

    return cos, sin

def gaussian_filter(da: xr.DataArray, **kwargs: float) -> xr.DataArray:
    """
    Apply a Gaussian filter to a `DataArray`.

    Parameters
    ----------
    da
        Array of data to filter.
    kwargs
        Keys should correspond to coordinates of `da`, and values should be the
        desired widths of the filter in that direction.

    Returns
    -------
    xr.DataArray
        Filtered array.

    """

    for name, width in kwargs.items():
        if name == 'z':
            name = [s for s in da.coords if s.startswith('z_')][0]

        if name in ['seconds', 'minutes', 'hours', 'days']:
            coord = cftime.date2num(da['time'], f'{name} since {EPOCH}')
            i = list(da.coords).index('time')

        else:
            coord = da[name]
            i = list(da.coords).index(name)

        sigma = max(1, int(width / abs(coord[1] - coord[0]) / 4))
        filtered = _filter(da.values, sigma, axis=i)
        da = xr.DataArray(filtered, da.coords)

    return da

def get_bump(z: np.ndarray, center: float, width: float) -> np.ndarray:
    """
    Calculate a Gaussian bump of unit amplitude on the provided grid.

    Parameters
    ----------
    z
        Grid points to on which to calculate the function.
    center, width
        Parameters of the Gaussian to compute.

    Returns
    -------
    np.ndarray
        Array of Gaussian curve values on `z`.

    """

    return np.exp(-0.5 * ((z - center) / width) ** 2)

def get_rho(z: np.ndarray) -> np.ndarray:
    """
    Return the background density profile as constant or decaying with height,
    depending on whether the Boussinesq approximation is made.

    Parameters
    ----------
    z
        Vertical grid points at which to calculate the density.

    Returns
    -------
    np.ndarray
        Density at each vertical grid point.

    """

    return config.rho_ref * np.exp(-z / config.H_rho)

def get_iterator() -> Iterator[int]:
    """
    Return a `tqdm` object configured to show useful integration output.

    Returns
    -------
    tqdm[int]
        Iterator from 1 to `config.n_steps`.

    """

    format = (
        '{desc}: {percentage:3.0f}%|' +
        '{bar}| {n:.2f}/{total_fmt} [{rate_fmt}{postfix}]'
    )

    return trange(
        1, config.n_steps,
        bar_format=format,
        unit_scale=(config.dt / 86400),
        unit='day'
    )

def get_time(dt: Optional[int]=None) -> np.ndarray:
    """
    Return an array of datetimes for each step in the integration.

    Parameters
    ----------
    dt
        Time step to use for each datetime. If `None`, use `config.dt`.

    Returns
    -------
    np.ndarray
        Array of datetimes.

    """

    if dt is None:
        dt = config.dt

    n_steps = n_steps = int(86400 * config.n_day / dt) + 1
    seconds = dt * np.arange(n_steps)

    return cftime.num2date(seconds, f'seconds since {EPOCH}')

def get_vertical_grids() -> tuple[np.ndarray, np.ndarray]:
    """
    Create the vertical grids of cell faces and cell centers. Provided as a
    utility so that codes can obtain grid information without instantiating an
    otherwise-unnecessary `MeanState` object.

    Returns
    -------
    np.ndarray
        Array of vertical grid cell faces.
    np.ndarray
        Array of vertical grid cell centers.

    """

    faces = np.linspace(config.z_min, config.z_max, config.n_grid)
    centers = (faces[:-1] + faces[1:]) / 2

    return faces, centers

def get_wavenumbers(u: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
    """
    Return data from the neural network input space (phase speed and intrinsic
    period) to wavenumber space.

    Parameters
    ----------
    u
        Tensor of wind profiles, as passed to the neural network.
    X_hat
        Tensor of phase speeds and intrinsic periods, as passed as the first two
        columns of the neural network input.

    Returns
    -------
    torch.Tensor, torch.Tensor
        Arrays of zonal and vertical wavenumbers, respectively.
    
    """

    cp_x, T_hat = output.T
    omega_hat = 2 * torch.pi / T_hat
    k = omega_hat / (cp_x - u[:, 0, 0])
    m = get_m(k, 0, omega_hat, config.N_ref)

    return k, m

def get_wind_input(mean: PrescribedWind, n_step: int) -> torch.Tensor:
    """
    Get the zonal wind inputs to a neural network.

    Parameters
    ----------
    mean
        Current mean state of the system.
    n_step
        Current time step.

    Returns
    -------
    torch.Tensor
        Tensor of zonal winds whose first dimension is a dummy, whose second
        dimension ranges over past snapshots, and whose third dimension
        ranges over vertical grid points.

    """

    steps = [n_step, max(n_step - config.lookback // config.dt, 0)]
    return torch.as_tensor(mean._wind[steps, 0])[None]

def make_colored_noise(
    xs: np.ndarray | list[np.ndarray],
    decay_scales: float | list[float],
    cutoff_scales: Optional[float | list[float]]=None,
    n_min: float=-1,
    n_max: float=1,
    rng: Optional[Generator]=None
) -> np.ndarray:
    """
    Generate power law noise in one or two dimensions. The first three arguments
    can be provided as lists or, if one-dimensional noise is to be generated, as 
    single arguments (that is, floats or numpy arrays). If `n_min == n_max`, the
    returned array will simply be constant with the appropriate shape.

    Parameters
    ----------
    xs
        Coordinate arrays.
    decay_scales
        Scales for each coordinate beyond which power law decay begins. The
        spectrum is more or less flat for larger scales.
    cutoff_scales
        Cutoff scales for each coordinate. Higher-frequency modes will have
        their amplitudes zeroed out. If `None` or a list of zeros, the noise
        will contain contributions from all resolvable modes.
    n_min
        Minimum value in returned noise.
    n_max
        Maximum value in returned noise.
    rng
        Optional random number generator to use in generating the random
        amplitudes, so that a call to this function can be reproducible without
        altering the global seed.

    Returns
    -------
    np.ndarray
        Array of noise with the appropriate shape and amplitude.
    
    """

    xs = _as_list(xs)
    decay_scales = _as_list(decay_scales)

    if cutoff_scales is None:
        cutoff_scales = [0] * len(xs)

    cutoff_scales = _as_list(cutoff_scales)

    if n_min == n_max:
        shape = tuple(len(x) for x in xs)
        return n_min * np.ones(shape)
    
    if len(xs) != len(decay_scales) != len(cutoff_scales) or len(xs) > 2:
        raise ValueError('Only one- or two-dimensional noise can be generated')
    
    dx = xs[0][1] - xs[0][0]
    k = np.fft.fftfreq(len(xs[0]), dx)
    decay = (k * decay_scales[0]) ** 2
    idx = abs(k) * cutoff_scales[0] > 1

    if len(xs) > 1:
        dy = xs[1][1] - xs[1][0]
        ell = np.fft.fftfreq(len(xs[1]), dy)
        decay = decay[:, None] + (ell * decay_scales[1]) ** 2
        idx = idx[:, None] | (abs(ell) * cutoff_scales[1] > 1)

    beta = 2 if len(xs) == 1 else 3
    alpha = 1 if len(xs) == 1 else 1 / 2
    power = 1 / (1 + alpha * np.sqrt(decay) ** beta)
    power[idx] = 0

    rng = rng if rng is not None else np.random.default_rng()
    fft_func = np.fft.ifft if len(xs) == 1 else np.fft.ifft2

    phase = 2 * np.pi * rng.random(power.shape)
    noise_hat = np.sqrt(power) * np.exp(1j * phase)
    noise = fft_func(noise_hat).real

    noise = (noise - noise.min()) / (noise.max() - noise.min())
    return n_min + noise * (n_max - n_min)

def open_dataset(*args, **kwargs) -> xr.Dataset:
    """
    Open a netCDF file as an xarray `Dataset`. This function exists so that
    use_cftime=True can be the default.

    Parameters
    ----------
    args, kwargs
        Arguments to be passed to `xr.open_dataset`.

    Returns
    -------
    xr.Dataset
        Dataset object opened using cftime.

    """

    return xr.open_dataset(*args, use_cftime=True, **kwargs)

def shapiro_filter(data: np.ndarray) -> np.ndarray:
    """
    Apply a zeroth-order Shapiro filter along the first axis.

    Parameters
    ----------
    data
        Array to be filtered.

    Returns
    -------
    np.ndarray
        Filtered array. Has two fewer elements along the first axis than `data`.

    """

    return (data[:-2] + 2 * data[1:-1] + data[2:]) / 4

def _as_list(a: _T | list[_T]) -> list[_T]:
    """
    Validate an argument that can be passed as a scalar or a list of scalars by
    ensuring that it is a list.

    Parameters
    ----------
    a
        Argument to validate.

    Returns
    -------
    list[_T]
        If `a` was a list, it is returned as-is. Otherwise, a list containing
        the passed scalar is returned.

    """

    if isinstance(a, list):
        return a
    
    return [a]
