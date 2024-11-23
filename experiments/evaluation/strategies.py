from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from msgwam import config
from msgwam.integration import integrate as _integrate
from msgwam.plotting import plot_time_series

from utils import get_rmse, load_flux

_COLORS = {
    'coarse' : 'royalblue',
    'instantaneous' : 'tab:red',
    'stochastic' : 'darkviolet'
}

_N_SAMPLES = 25

def integrate(strategy: str) -> None:
    """
    Integrate using a particular strategy, and save the results.

    Parameters
    ----------
    strategy
        Name of the strategy to use. There should be a function in this module
        with a name of the form `_get_{strategy}_kwargs` that returns the
        configuration overrides for the strategy.

    """

    func_name = f'_get_{strategy}_kwargs'
    kwargs = globals()[func_name]()

    with config.override(**kwargs):
        if strategy != 'stochastic':
            ds = _integrate()

        else:
            datasets = []
            for i in range(_N_SAMPLES):
                ds = _integrate().assign_coords(sample=np.array([i]))
                datasets.append(ds)

            ds = xr.concat(datasets, dim='sample')
        
        ds.to_netcdf(f'data/{config.name}/{strategy}.nc')

def plot_errors() -> None:
    """
    Plot the root-mean-square of each strategy with respect to the reference as
    a function of height. Also plot the errors in root-mean-square fluxes, to
    get a sense of where fluxes are over- or underestimated.
    """

    z_faces = np.linspace(config.z_min, config.z_max, config.n_grid)
    ref = load_flux(f'data/{config.name}/reference.nc', z_faces)
    rms = get_rmse(ref)

    fig, axes = plt.subplots(ncols=2)
    fig.set_size_inches(6, 4.5)

    z_plot = z_faces / 1000
    for strategy, color in _COLORS.items():
        pmf = load_flux(f'data/{config.name}/{strategy}.nc', z_faces)
        if strategy == 'stochastic':
            pmf = pmf.mean('sample')

        rmse = 1000 * get_rmse(pmf, ref)
        erms = 1000 * (get_rmse(pmf) - rms)
        axes[0].plot(rmse, z_plot, color=color, label=strategy)
        axes[1].plot(erms, z_plot, color=color)

    label = r'$\langle F_{\mathrm{ref}} \rangle$'
    axes[0].plot(1000 * rms, z_plot, color='gray', ls='dashed', label=label)
    axes[0].legend()

    axes[0].set_xlim(0, 1.5)
    axes[1].set_xlim(-0.5, 0.5)

    axes[0].set_xlabel(r'$\langle F - F_{\mathrm{ref}} \rangle$ (mPa)')
    axes[1].set_xlabel(
        r'$\langle F \rangle - \langle F_{\mathrm{ref}} \rangle$ (mPa)'
    )

    for ax in axes:
        ax.set_ylim(z_plot.min(), z_plot.max())
        ax.set_ylabel('height (km)')

        ax.grid(color='lightgray')
        ax.tick_params('both', direction='in')

    plt.tight_layout()
    plt.savefig(f'plots/{config.name}/errors.png', dpi=400)

def plot_fluxes() -> None:
    """Plot the flux time series for each strategy."""

    z_faces = np.linspace(config.z_min, config.z_max, config.n_grid)
    strategies = ['reference'] + list(_COLORS.keys())
    n_rows = (len(strategies) - 1) // 3 + 1
    n_cols = min(3, len(strategies))

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols)
    fig.set_size_inches(4.5 * n_cols, 3 * n_rows)
    axes = axes.flatten()

    for i, (strategy, ax) in enumerate(zip(strategies, axes)):
        pmf = load_flux(f'data/{config.name}/{strategy}.nc', z_faces)
        if strategy == 'stochastic':
            pmf = pmf.mean('sample')

        plot_time_series(1000 * pmf, 3, [ax])
        ax.set_title(strategy)

        if i % 3 != 0:
            ax.set_ylabel(None)

    plt.tight_layout()
    plt.savefig(f'plots/{config.name}/fluxes.png', dpi=400)

def _get_coarse_kwargs() -> dict[str, Any]:
    """
    Once the configuration file has been updated as in `coarsening.py`, the
    coarse strategy simply uses those settings as is.
    """

    return {}

def _get_instantaneous_kwargs() -> dict[str, Any]:
    """
    Use the steady-state monochromatic propagator instead of the ray tracer.
    """

    return {'propagator_type' : 'instantaneous'}

def _get_stochastic_kwargs() -> dict[str, Any]:
    """
    Use a phase space resolution that is ten times finer, but launches rays ten
    times less frequently.
    """

    return {
        'epsilon' : 1 / 9,
        'dr_init' : config.dr_init / 3,
        'n_source' : int(config.n_source * 3),
        'source_type' : 'stochastic'
    }
