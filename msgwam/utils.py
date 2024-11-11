from abc import ABC
from itertools import takewhile
from typing import Iterator, Self

import cftime
import numpy as np
import xarray as xr

from tqdm import trange

from . import config
from .constants import EPOCH

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
    time: np.ndarray,
    z: np.ndarray,
    T: float,
    H: float,
    beta: float=2
) -> np.ndarray:
    """
    Generate power law noise in time and height.

    Parameters
    ----------
    time
        Grid of time steps.
    z
        Grid of vertical grid levels.
    T
        Period of the dominant mode in time. The e-folding time of the temporal
        autocorrelation will be `T / (2 * pi)`.
    H
        Length scale of the dominant mode in the vertical. The e-folding time of
        the height autocorrelation will be `H / (2 * pi)`.
    beta
        Decay rate of frequencies above the critical frequency. This parameter
        will describe the power spectrum along cross-sections in time or height,
        so that e.g. `beta=2` gives classical red noise.

    Returns
    -------
    np.ndarray
        Two-dimensional array of noise, normalized ot lie between -1 and 1.
    
    """

    dt = time[1] - time[0]
    dz = z[1] - z[0]

    k = np.fft.fftfreq(len(time), dt)[:, None]
    ell = np.fft.fftfreq(len(z), dz)

    k_cutoff = 1 / T
    ell_cutoff = 1 / H
    decay = (k / k_cutoff) ** 2 + (ell / ell_cutoff) ** 2
    power = 1 / (1 + 0.5 * np.sqrt(decay) ** (2 * beta - 1))

    phase = 2 * np.pi * np.random.rand(*power.shape)
    noise_hat = np.sqrt(power) * (np.cos(phase) + 1j * np.sin(phase))
    noise = np.fft.ifft2(noise_hat).real

    return 2 * (noise - noise.min()) / (noise.max() - noise.min()) - 1

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