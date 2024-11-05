import numpy as np

from msgwam import config
from msgwam.means import PrescribedWind
from msgwam.propagators import TransientPropagator

def get_min_dr(round_to: int=25, **kwargs) -> float:
    """
    Calculate the minimum dr value that can be resolved by the bottom boundary
    condition, given some configuration overrides.

    Parameters
    ----------
    kwargs
        Settings to pass to `config.override`.

    Returns
    -------
    float
        Minimum resolvable dr.

    """

    with config.override(**kwargs):
        mean = PrescribedWind()
        cg = TransientPropagator(mean)._get_cg_r(mean)
        distance = np.nanmax(cg * config.dt)

    return np.ceil(distance / round_to) * round_to

def make_colored_noise(n_t: int, n_z: int, p: float=(5 / 3)) -> np.ndarray:
    """
    Generate power law noise in time and height. The grid need not have the same
    number of points in each dimension.

    Parameters
    ----------
    n_t
        Number of time steps.
    n_z
        Number of vertical levels.
    p
        Power governing amplitude decay.

    Returns
    -------
    np.ndarray
        Two-dimensional array of noise. Normalized to lie between -1 and 1.

    """

    ell = n_z * np.fft.fftfreq(n_z)
    k = n_t * np.fft.fftfreq(n_t)[:, None]
    wvn_sq = (k ** 2 + ell ** 2) / (n_t ** 2 + n_z ** 2)

    idx = wvn_sq != 0
    A = np.zeros_like(wvn_sq)
    A[idx] = 1 / wvn_sq[idx]
    A = A ** (p / 2)

    phase = 2 * np.pi * np.random.rand(*A.shape)
    noise_hat = A * (np.cos(phase) + 1j * np.sin(phase))
    noise = np.fft.ifft2(noise_hat).real

    return 2 * (noise - noise.min()) / (noise.max() - noise.min()) - 1