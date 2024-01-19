import sys
sys.path.insert(0, '.')

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from msgwam import config, integrate
from msgwam.mean import MeanFlow
from msgwam.plotting import make_plots
from msgwam.rays import RayCollection

COLORS = ['k', 'tab:red', 'royalblue', 'forestgreen', 'darkviolet']
PREFIX = 'data/baselines'

SPEEDUP = 9
SPINUP = 2

def plot_source_spectrum() -> None:
    rays = RayCollection(MeanFlow())
    source = rays.sources
    rays.data = source

    k = rays.k
    l = rays.l
    m = rays.m
    dens = rays.dens

    cg_r = rays.cg_r()
    volume = abs(rays.dk * rays.dl * rays.dm)
    flux = 1000 * rays.k * dens * volume * cg_r
    print(f'total flux: {abs(flux).sum():.2f} mPa')

    c = np.sign(k) * np.sqrt(
        (config.N0 ** 2 * (k ** 2 + l ** 2) + config.f0 ** 2 * m ** 2) /
        (k ** 2 * (k ** 2 + l ** 2 + m ** 2))
    )

    plt.xlabel('phase speed (m / s)')
    plt.ylabel('momentum flux (mPa)')

    plt.xticks(np.linspace(-50, 50, 11))
    plt.xlim(-50, 50)
    
    plt.grid(True, color='lightgray', zorder=0)
    plt.plot(c[c < 0], flux[c < 0], marker='o', color='k', zorder=5)
    plt.plot(c[c > 0], flux[c > 0], marker='o', color='k', zorder=5)

    plt.tight_layout()
    plt.savefig('plots/baselines/source.png', dpi=400)
    plt.close()

def save_reference_mean_state(truncate: bool=True) -> None:
    ds = integrate.SBDF2Integrator().integrate().to_dataset()
    if truncate:
        ds = ds.sel(time=slice(75 * 86400, None))
        ds['time'] = ds['time'] - ds['time'][0]

    make_plots(ds, f'plots/baselines/mean-wind.png')
    ds.to_netcdf(f'{PREFIX}/mean-wind.nc')

def _run_refinement(name: str) -> None:
    for flag, suffix in zip([True, False], ['int', 'fix']):
        config.interactive_mean = flag
        path = f'{PREFIX}/{name}-{suffix}.nc'
        print(f'{name}-{suffix}')

        config.refresh()
        ds = integrate.SBDF2Integrator().integrate().to_dataset()
        make_plots(ds, f'plots/baselines/{name}-{suffix}.png')
        ds.to_netcdf(path)

def run_refinements() -> None:
    _run_refinement('reference')

    config.n_ray_max = config.n_ray_max // SPEEDUP
    _run_refinement('fine')

    config.epsilon = 1 / SPEEDUP
    _run_refinement('stochastic')
    config.epsilon = 1

    drs = config.dr_init * np.array([1, SPEEDUP, np.sqrt(SPEEDUP)])
    n_sources = config.n_source / np.array([SPEEDUP, 1, np.sqrt(SPEEDUP)])

    n_sources = n_sources.astype(int)
    n_sources[np.mod(n_sources, 2) == 1] += 1    
    shapes = ['wide', 'tall', 'square']

    for dr, n_source, shape in zip(drs, n_sources, shapes):
        config.dr_init = dr
        config.n_source = n_source
        _run_refinement(f'coarse-{shape}')

def _get_error(profile: np.ndarray, ref: np.ndarray, axis: int=0) -> np.ndarray:
    # scale = abs(ref).mean(axis=axis)
    # error = np.sqrt(((profile - ref) ** 2).mean(axis=axis))
    # return error / scale

    return np.sqrt(((profile - ref) ** 2).mean(axis=axis))

def make_summary_plot(plot_by='height'):
    references = {}
    with xr.open_dataset(f'{PREFIX}/reference-fix.nc') as ds:
        ds = ds.sel(time=slice(SPINUP * 86400, None))

        days = ds['time'].values / 86400
        references['fix'] = ds['pmf_u'].values * 1000
        z = ds['grid'].values / 1000

    with xr.open_dataset(f'{PREFIX}/reference-int.nc') as ds:
        ds = ds.sel(time=slice(SPINUP * 86400, None))

        references['int'] = ds['pmf_u'].values * 1000
        references['u'] = ds['u'].values
        runtime: float = ds.runtime
        
    shapes = ['wide', 'tall', 'square']
    kinds = ['fine'] + [f'coarse-{shape}' for shape in shapes] + ['stochastic']

    dims = {'height' : (13.5, 6), 'time' : (18, 4.5)}[plot_by]
    fig, axes = plt.subplots(ncols=3)
    fig.set_size_inches(*dims)

    axis = ['height', 'time'].index(plot_by)
    for i, suffix in enumerate(['fix', 'int']):
        for kind, color in zip(kinds, COLORS):
            with xr.open_dataset(f'{PREFIX}/{kind}-{suffix}.nc') as ds:
                ds = ds.sel(time=slice(SPINUP * 86400, None))

                pmf = ds['pmf_u'].values * 1000
                percent = (ds.runtime / runtime) * 100

            error = _get_error(pmf, references[suffix], axis)
            kind_name = kind.replace('-', ', ').replace('fine', 'do-nothing')
            label = kind_name + f' ({percent:.1f}% runtime)'

            if plot_by == 'height':
                axes[i].plot(error, z, color=color, label=label)
            elif plot_by == 'time':
                axes[i].plot(days, error, color=color, label=label)

        guide = abs(references[suffix]).mean(axis=axis)
        label = '$\overline{|\mathrm{reference}|}$'

        if plot_by == 'height':
            axes[i].plot(guide, z, color='gray', ls='dashed', label=label)
        elif plot_by == 'time':
            axes[i].plot(days, guide, color='gray', ls='dashed', label=label)

    for kind, color in zip(kinds, COLORS):
        with xr.open_dataset(f'{PREFIX}/{kind}-int.nc') as ds:
            ds = ds.sel(time=slice(SPINUP * 86400, None))

            u = ds['u'].values

        error = _get_error(u, references['u'], axis)

        if plot_by == 'height':
            axes[2].plot(error, z, color=color)
        elif plot_by == 'time':
            axes[2].plot(days, error, color=color)

    guideline = abs(references['u']).mean(axis=axis)
    if plot_by == 'height':
        axes[2].plot(guideline, z, color='gray', ls='dashed')
    elif plot_by == 'time':
        axes[2].plot(days, guideline, color='gray', ls='dashed')

    if plot_by == 'height':
        for i in range(2):
            axes[i].set_xlim(0, 0.8)
            axes[i].set_xticks(np.linspace(0, 0.8, 5))
            axes[i].set_xlabel('RMSE (mPa)')

        axes[2].set_xlim(0, 15)
        axes[2].set_xlabel('RMSE (m / s)')
        axes[2].set_xticks(np.linspace(0, 15, 6))

        for ax in axes:
            ax.set_ylim(0, 100)
            ax.set_ylabel('height (km)')

            ax.grid(True, color='lightgray')

    elif plot_by == 'time':
        for i in range(2):
            axes[i].set_ylim(0, 0.8)
            axes[i].set_yticks(np.linspace(0, 0.8, 5))
            axes[i].set_ylabel('RMSE (mPa)')

        axes[2].set_ylim(0, 15)
        axes[2].set_ylabel('RMSE (m / s)')
        axes[2].set_yticks(np.linspace(0, 15, 6))

        for ax in axes:
            ax.set_xlim(SPINUP, days.max())
            ax.set_xlabel('time (days)')

            ax.grid(True, color='lightgray')

    axes[1].legend()
    axes[0].set_title('MF error with fixed $u$')
    axes[1].set_title('MF error with interactive $u$')
    axes[2].set_title('interactive $u$ error')
    
    plt.tight_layout()
    plt.savefig(f'plots/baselines/scores-by-{plot_by}.png', dpi=400)

if __name__ == '__main__':
    config.load_config('config/baselines.toml')
    plot_source_spectrum()
    save_reference_mean_state(True)

    config.nu = 1
    config.n_day = 50
    config.bc_mom_flux = 3e-3
    config.refresh()

    run_refinements()
    make_summary_plot('height')
    make_summary_plot('time')
