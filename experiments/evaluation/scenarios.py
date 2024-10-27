import cftime
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from msgwam import config
from msgwam.constants import EPOCH
from msgwam.means import InteractiveWind
from msgwam.plotting import plot_time_series
from msgwam.utils import shapiro_filter

def save_mean_state() -> None:
    """Save a mean state file for this configuration."""

    np.random.seed(1234)
    ds = _descending_jets()

    _, cbar = plot_time_series(ds['u'], 50, cmap='PuOr_r')
    cbar.set_label('$\\bar{u}$ (m / s)')

    plt.tight_layout()
    plt.savefig(f'plots/{config.name}/mean-state.png', dpi=400)
    ds.to_netcdf(f'data/{config.name}/mean-state.nc')

def _descending_jets() -> xr.Dataset:
    """
    Generate a mean flow scenario with descending jets approximating the QBO.

    Returns
    -------
    xr.Dataset
        Dataset containing both components of the mean wind.

    """

    seconds = config.dt * np.arange(config.n_steps)
    time = cftime.num2date(seconds, f'seconds since {EPOCH}')
    z = InteractiveWind().z_centers

    k = 2 * np.pi / (0.5 * 86400)
    ell = 2 * np.pi / 25e3

    x, y = np.meshgrid(seconds, z)
    wave = np.exp(1j * (k * x + ell * y)).real.T
    env = np.exp(-((z - 45e3) / 10e3) ** 2)

    args = [config.n_steps, config.n_grid - 1]
    noise_1 = _make_colored_noise(*args)
    noise_2 = _make_colored_noise(*args)

    u = 30 * env * (wave + noise_1) + 5 * noise_2
    u[:, 1:-1] = shapiro_filter(u.T).T
    v = np.zeros_like(u)

    data = {'time' : time, 'z_centers' : z}
    data['u'] = (('time', 'z_centers'), u)
    data['v'] = (('time', 'z_centers'), v)

    return xr.Dataset(data)

def _make_colored_noise(n_t: int, n_z: int, p: float=(5 / 3)) -> np.ndarray:
    """
    Generate power law noise on a potentially non-square time-height grid.

    Parameters
    ----------
    n_t
        Number of points in the time dimension.
    n_z
        Number of points in the height dimension.
    p
        Power governing amplitude decay.

    Returns
    -------
    np.ndarray
        Two-dimensional array of noise, ranging from -1 to 1.

    """

    ell = n_z * np.fft.fftfreq(n_z)
    k = n_t * np.fft.fftfreq(n_t)[:, None]
    wvn_sq = (k ** 2 + ell ** 2) / (n_t ** 2 + n_z ** 2)

    A = np.zeros_like(wvn_sq)
    A[wvn_sq != 0] = 1 / wvn_sq[wvn_sq != 0]
    A = A ** (p / 2)

    phase = 2 * np.pi * np.random.rand(*A.shape)
    noise_hat = A * (np.cos(phase) + 1j * np.sin(phase))
    noise = np.fft.ifft2(noise_hat).real

    return 2 * (noise - noise.min()) / (noise.max() - noise.min()) - 1