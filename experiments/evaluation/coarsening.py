from itertools import product

import matplotlib.gridspec as gs
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import Rectangle

from msgwam import config
from msgwam.integration import integrate

from plotting import plot_error_grid, plot_rmse_profiles
from utils import get_rmse, load_flux

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
    
    to_plot = []
    colors = ['forestgreen', 'royalblue', 'gray']
    linestyles = ['solid'] * 2 + ['dashed']

    for func, color in zip([np.argmin, np.argmax], colors):
        i, j = np.unravel_index(func(errors), errors.shape)
        to_plot.append(profiles[i, j])

        axes[1].add_patch(Rectangle(
            (i - 0.5, j - 0.5), 1, 1,
            ec=color, fc='none',
            clip_on=False,
            linewidth=2,
            zorder=10
        ))

    to_plot.append(rms)
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
            fname = f'dr-{dr}_n-source-{n_source}'
            integrate().to_netcdf(f'data/{config.name}/coarse/{fname}.nc')

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

    drs = [500 * i for i in range(1, 11)]
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
    profiles = np.zeros((len(drs), len(n_sources), len(z_faces)))

    args = [z_faces, '3h', config.n_day - 10]
    ref = load_flux(f'data/{config.name}/reference.nc', *args)
    
    for i, dr in enumerate(drs):
        for j, n_source in enumerate(n_sources):
            fname = f'dr-{dr}_n-source-{n_source}'
            flux = load_flux(f'data/{config.name}/coarse/{fname}.nc', *args)

            profiles[i, j] = get_rmse(ref, flux)

    rms = get_rmse(ref).values
    errors = (profiles / rms).mean(axis=-1)

    return profiles, errors, rms
