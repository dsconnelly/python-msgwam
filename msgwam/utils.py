from abc import ABC
from itertools import takewhile
from typing import Iterator, Optional, Self, TypeVar

import cftime
import numpy as np
import xarray as xr

from tqdm import trange

from . import config
from .constants import EPOCH

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

        is_lower = lambda c: c.islower()
        handle = lambda s: s[0].lower() + ''.join(takewhile(is_lower, s[1:]))
        subs = {handle(sub.__name__) : sub for sub in cls.__subclasses__()}

        return subs[name](*args, **kwargs)

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

def get_time() -> np.ndarray:
    """
    Return an array of datetimes for each step in the integration.

    Returns
    -------
    np.ndarray
        Array of datetimes.

    """

    seconds = config.dt * np.arange(config.n_steps)
    return cftime.num2date(seconds, f'seconds since {EPOCH}')

def make_colored_noise(
    xs: np.ndarray | list[np.ndarray],
    decay_scales: float | list[float],
    cutoff_scales: Optional[float | list[float]]=None,
    n_min: float=-1,
    n_max: float=1
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

    func = np.fft.ifft if len(xs) == 1 else np.fft.ifft2
    phase = 2 * np.pi * np.random.rand(*power.shape)
    noise_hat = np.sqrt(power) * np.exp(1j * phase)
    noise = func(noise_hat).real

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
