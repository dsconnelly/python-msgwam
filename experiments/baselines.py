import os
import sys
sys.path.insert(0, '.')

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from msgwam import config, integrate
from msgwam.plotting import plot_integration, plot_source

N_HOURS = 12
SPEEDUP = 9
TDX = slice(2 * 86400, None)

COLORS = {
    'do-nothing' : 'black',
    'stochastic' : 'darkviolet',
    'smoothed' : 'gold',
    'coarse-square' : 'royalblue',
    'coarse-tall' : 'forestgreen',
    'coarse-wide' : 'tab:red'
}

def save_mean_wind() -> None:
    ds = integrate.SBDF2Integrator().integrate().to_dataset()
    ds = ds.sel(time=slice((config.n_day - 30) * 86400, None))
    ds['time'] = ds['time'] - ds['time'][0]

    plot_integration(ds, 'plots/baselines/mean-wind.png')
    ds.to_netcdf('data/baselines/mean-wind.nc')

def run(name: str) -> None:
    print('=' * 16, name, '=' * 16)
    for flag, suffix in zip([False, True], ['fix', 'int']):
        config.interactive_mean = flag
        config.refresh()

        ds = integrate.SBDF2Integrator().integrate().to_dataset()
        plot_integration(ds, f'plots/baselines/{name}-{suffix}.png')
        ds.to_netcdf(f'data/baselines/{name}-{suffix}.nc')

def run_baselines() -> None:
    config.epsilon = 1 / SPEEDUP
    run('stochastic')

    config.epsilon = 1
    run('do-nothing')

def run_smoothings() -> None:
    config.shapiro_filter = False
    config.proj_method = 'gaussian'
    config.smoothing = 5
    config.tau = 3 * 3600

    run('smoothed')

    config.shapiro_filter = True
    config.proj_method = 'discrete'
    config.smoothing = 1
    config.tau = 0

def run_coarsenings() -> None:
    shapes = ['wide', 'tall', 'square']
    drs = config.dr_init * np.array([1, SPEEDUP, np.sqrt(SPEEDUP)])
    n_sources = config.n_source / np.array([SPEEDUP, 1, np.sqrt(SPEEDUP)])
    n_sources = n_sources.astype(int)

    for dr, n_source, shape in zip(drs, n_sources, shapes):
        config.dr_init = dr
        config.n_source = n_source
        run(f'coarse-{shape}')

def plot_scores() -> None:
    references = {}
    for suffix in ['fix', 'int']:
        with xr.open_dataset(f'data/baselines/reference-{suffix}.nc') as ds:
            ds = ds.sel(time=TDX)
            z = ds['grid'].values / 1000

            references[suffix] = ds['pmf_u'] * 1000
            runtime: float = ds.runtime

            if suffix == 'int':
                references['u'] = ds['u']

    fnames = os.listdir('data/baselines')
    kinds = {'-'.join(fname.split('-')[:-1]) for fname in fnames}
    kinds = kinds.difference({'mean', 'reference'})

    fig, axes = plt.subplots(ncols=3)
    fig.set_size_inches(13.5, 6)

    for i, suffix in enumerate(['fix', 'int']):
        for kind, color in COLORS.items():
            with xr.open_dataset(f'data/baselines/{kind}-{suffix}.nc') as ds:
                ds = ds.sel(time=TDX)
                pmf = ds['pmf_u'] * 1000
                percent = (ds.runtime / runtime) * 100

            error = get_error(pmf, references[suffix], N_HOURS)
            label = format_name(kind) + f' ({percent:.1f}% runtime)'
            axes[i].plot(error, z, color=color, label=label)

    for kind, color in COLORS.items():
        with xr.open_dataset(f'data/baselines/{kind}-int.nc') as ds:
            ds = ds.sel(time=TDX)
            u = ds['u']

        error = get_error(u, references['u'], N_HOURS)
        axes[2].plot(error, z, color=color)

    for i, suffix in enumerate(['fix', 'int', 'u']):
        guide = abs(references[suffix]).mean(axis=0)
        label = '$\overline{|\mathrm{reference}|}$'
        axes[i].plot(guide, z, color='gray', ls='dashed', label=label)

    for i, ax in enumerate(axes):
        ax.set_ylim(0, 60)
        ax.set_ylabel('height (km)')
        ax.grid(True, color='lightgray')

        if i < 2:
            ax.set_xlim(0, 0.5)
            ax.set_xticks(np.linspace(0, 0.5, 6))
            ax.set_xlabel('RMSE (mPa)')

        else:
            ax.set_xlim(0, 25)
            ax.set_xticks(np.linspace(0, 25, 6))
            ax.set_xlabel('RMSE (m / s)')

    axes[1].legend()
    axes[0].set_title('MF error with fixed $u$')
    axes[1].set_title('MF error with interactive $u$')

    plt.tight_layout()
    plt.savefig('plots/baselines/scores.png', dpi=400)

def plot_lifetimes() -> None:
    fig, axes = plt.subplots(ncols=2)
    fig.set_size_inches(12, 4.5)

    kinds = ['reference'] + list(COLORS.keys())
    labels = list(map(format_name, kinds))

    lifetimes, heights = [], []
    for kind in kinds:
        with xr.open_dataset(f'data/baselines/{kind}-fix.nc') as ds:
            ds = ds.sel(time=TDX)
            age = ds['age'].values
            meta = ds['meta'].values
            r = ds['r'].values

            changed = meta[:-1] != meta[1:]
            lifetimes.append(age[:-1][changed].mean())
            heights.append(r[:-1][changed].mean())

    x = np.arange(len(kinds))
    lifetimes = np.array(lifetimes) / 60
    heights = np.array(heights) / 1000

    axes[0].bar(x, lifetimes, width=0.8, color='lightgray', edgecolor='k')
    axes[1].bar(x, heights, width=0.8, color='lightgray', edgecolor='k')

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45)
        ax.set_xlim(-0.5, len(kinds) - 0.5)

    axes[0].set_ylabel('average lifetime (m)')
    axes[1].set_ylabel('average maximum height (km)')

    plt.tight_layout()
    plt.savefig('plots/baselines/lifetimes.png', dpi=400)

def format_name(kind: str) -> str:
    return kind.replace('-', ' ').replace('coarse ', 'coarse, ')

def get_error(
    profile: xr.DataArray,
    ref: xr.DataArray,
    n_hours: Optional[int]=None
) -> np.ndarray:
    if n_hours is not None:
        bins = np.round(profile['time'] / (n_hours * 3600))
        profile = profile.groupby(bins).mean()
        ref = ref.groupby(bins).mean()

    return np.sqrt(((profile - ref) ** 2).mean('time'))
    
if __name__ == '__main__':
    args = sys.argv[1:]
    config.load_config('config/baselines.toml')

    if 'plot-source' in args:
        plot_source('plots/baselines/source.png')

    if 'save-mean' in args:
        save_mean_wind()

    config.n_day = 25
    config.dt_output = 120
    config.n_ray_max = 2475

    config.proj_method = 'discrete'
    config.shapiro_filter = True
    config.tau = 0

    config.mu = 1e-3
    config.bc_mom_flux = 3e-3
    config.dissipation = False

    if 'run' in args:
        run('reference')

        config.n_ray_max = config.n_ray_max // SPEEDUP

        # run_baselines()
        # run_smoothings()
        # run_coarsenings()

    if 'plot' in args:
        # plot_scores()
        plot_lifetimes()
