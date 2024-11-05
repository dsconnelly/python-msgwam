from itertools import product

import matplotlib.gridspec as gs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from matplotlib.patches import Rectangle

from msgwam import config
from msgwam.integration import integrate
from msgwam.utils import open_dataset

from plotting import plot_error_grid, plot_rmse_profiles
from utils import get_min_dr

def plot_coarse_errors() -> None:
    """
    Make a plot of the of the normalized errors in each coarse configuration.
    Also plot the RMSE profiles of the extremal configurations.
    """

    widths = [3, 4.5, 0.2]
    fig = plt.figure(constrained_layout=True)
    fig.set_size_inches(sum(widths), 4.5)

    spec = gs.GridSpec(1, 3, figure=fig, width_ratios=widths)
    axes = [fig.add_subplot(spec[0, j]) for j in range(2)]
    cax = fig.add_subplot(spec[0, 2])

    drs, n_sources = _get_grid()
    profiles, errors, rms = _get_errors(drs, n_sources)
    plot_error_grid(drs, n_sources, errors, (axes[1], cax))
    
    to_plot = [rms]
    colors = ['gray', 'forestgreen', 'royalblue']
    linestyles = ['dashed'] + ['solid'] * 2

    for func, color in zip([np.argmin, np.argmax], colors[1:]):
        i, j = np.unravel_index(func(errors), errors.shape)
        to_plot.append(profiles[i, j])

        axes[1].add_patch(Rectangle(
            (i - 0.5, j - 0.5), 1, 1,
            ec=color, fc='none',
            clip_on=False,
            linewidth=2,
            zorder=10
        ))

    z = np.linspace(config.z_min, config.z_max, config.n_grid) / 1000
    plot_rmse_profiles(z, to_plot, colors, linestyles, ax=axes[0])

    plt.savefig(f'plots/{config.name}/coarse-errors.png', dpi=400)

def save_coarsenings() -> None:
    """
    Integrate with a grid of values for vertical and spectral resolution and low
    `config.n_max`, saving the output of each configuration.
    """

    drs, n_sources = _get_grid()
    for dr, n_source in product(drs, n_sources):
        with config.override(dr_init=float(dr), n_source=n_source):
            fname = f'coarse_dr-{dr}_n-source-{n_source}'
            integrate().to_netcdf(f'data/{config.name}/{fname}.nc')

def update_config() -> None:
    """
    Update the values of `dr_init` and `n_source` in the loaded configuration
    file to use the best values found during the grid search.
    """

    drs, n_sources = _get_grid()
    _, errors, _ = _get_errors(drs, n_sources)
    i, j = np.unravel_index(np.argmin(errors), errors.shape)

    with open(f'config/{config.name}.toml') as f:
        lines = f.readlines()

    with open(f'config/{config.name}.toml', 'w') as f:
        for line in lines:
            if line.startswith('dr_init'):
                f.write(f'dr_init = {drs[i]}\n')

            elif line.startswith('n_source'):
                f.write(f'n_source = {n_sources[j]}\n')

            else:
                f.write(line)

def _get_grid() -> tuple[list[int], list[int]]:
    """
    Return lists of values for `config.dr_init` and `config.n_source` to use
    in the low `config.n_max` integrations.

    Returns
    -------
    list[int]
        Values for `config.dr`.
    list[int]
        Values for `config.n_source`.

    """

    min_dr = get_min_dr()
    drs = [int(min_dr + 500 * i) for i in range(10)]
    n_sources = [10 * i for i in range(1, 11)]

    return drs, n_sources[::-1]

def _get_errors(
    drs: list[int],
    n_sources: list[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute root-mean-square errors as a function of height and normalized error
    across all coarsened configurations. For convenience, also returns the RMS
    flux as a function of height in the reference integration.

    Parameters
    ----------
    drs
        Values of `config.dr_init`.
    n_sources
        Values of `config.n_source`.

    Returns
    -------
    np.ndarray
        Two-dimensional grid of root-mean-square error profiles whose last
        dimension corresponds to height.
    np.ndarray
        Two-dimensional grid of normalized errors averaged over all heights.
    np.ndarray
        RMS flux in the reference integration.

    """

    
    z_faces = np.linspace(config.z_min, config.z_max, config.n_grid)
    ref = _load_flux(f'data/{config.name}/reference.nc', z_faces)
    profiles = np.zeros((len(drs), len(n_sources), len(z_faces)))

    for i, dr in enumerate(drs):
        for j, n_source in enumerate(n_sources):
            fname = f'coarse_dr-{dr}_n-source-{n_source}'
            flux = _load_flux(f'data/{config.name}/{fname}.nc', z_faces)
            profiles[i, j] = _get_rmse(ref, flux)

    rms = _get_rmse(ref).values
    errors = (profiles / rms).mean(axis=-1)

    return profiles, errors, rms

def _get_rmse(a: xr.DataArray, b: xr.DataArray | float = 0) -> xr.DataArray:
    """
    Compute the root-mean-square error over time between two arrays. The second
    argument can also be passed in as a constant float, so that this function
    can be used to calculate the RMS value of the data by passing in zero.

    Parameters
    ----------
    a, b
        Data with which to compute RMS errors.

    Returns
    -------
    xr.DataArray
        Array of RMS errors, with the time dimension averaged out.

    """

    return np.sqrt(((a - b) ** 2).mean('time'))

def _load_flux(path: str, z_faces: np.ndarray) -> xr.DataArray:
    """
    Load the zonal gravity wave momentum flux from a netCDF file. This function
    adds the easterly and westerly components and ensures that the returned flux
    is at the correct temporal and vertical resolution.

    Parameters
    ----------
    path
        Path to netCDF file containing flux data.
    z_faces
        Array of vertical grid faces to interpolate onto.

    Returns
    -------
    xr.DataArray
        Postprocessed flux time series.

    """

    with open_dataset(path) as ds:
        ds = ds.interp(z_faces=z_faces)
        ds = ds.resample(time='1h').mean('time')
        flux = ds['pmf_e'] + ds['pmf_w']

    return flux