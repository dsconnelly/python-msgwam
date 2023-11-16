import sys
sys.path.insert(0, '.')

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from msgwam import config, integrate

def save_reference_mean_state():
    ds = integrate.SBDF2Integrator().integrate().to_dataset()
    ds = ds.sel(time=slice(25 * 86400, 40 * 86400))
    ds['time'] = ds['time'] - ds['time'][0]
    ds.to_netcdf('data/baseline-mean.nc')

def run_refinements():
    factor = int(1 / config.source_fraction)

    ds = integrate.SBDF2Integrator().integrate().to_dataset()
    ds.to_netcdf(f'data/baseline-reference.nc')

    config.n_ray_max = config.n_ray_max // factor
    config.source_type = 'stochastic'
    config.refresh()

    ds = integrate.SBDF2Integrator().integrate().to_dataset()
    ds.to_netcdf(f'data/baseline-stochastic.nc')

    drs = config.dr_init * np.array([1, factor, np.sqrt(factor)])
    dms = config.dm_init * np.array([factor, 1, np.sqrt(factor)])
    suffixes = ['wide', 'tall', 'square']

    config.source_type = 'constant'
    for dr, dm, suffix in zip(drs, dms, suffixes):
        config.dr_init = dr
        config.dm_init = dm
        config.refresh()

        ds = integrate.SBDF2Integrator().integrate().to_dataset()
        ds.to_netcdf(f'data/baseline-coarse-{suffix}.nc')

def make_summary_plot():
    fig, axes = plt.subplots(ncols=2)
    fig.set_size_inches(12, 4,5)

    with xr.open_dataset('data/baseline-mean.nc') as ds:
        days = ds['time'].values / (60 * 60 * 24)
        grid = ds['grid'] / 1000

        u = ds['u'].values
        amax = abs(u).max()

        axes[0].pcolormesh(
            days,
            grid,
            u.T,
            vmin=-amax,
            vmax=amax,
            cmap='RdBu_r',
            shading='nearest'
        )
        
    axes[0].set_title('fixed mean wind')
    axes[0].set_xlabel('time (days)')
    axes[0].set_ylabel('height (km)')

    axes[0].set_xlim(0, days.max())
    axes[0].set_ylim(0, 100)
    axes[0].set_yticks(np.linspace(0, 100, 6))

    with xr.open_dataset('data/baseline-reference.nc') as ds:
        ref = ds['pmf_u'].values * 1000

    colors = ['tab:red', 'royalblue', 'forestgreen', 'darkviolet']
    suffixes = [f'coarse-{kind}' for kind in ['wide', 'tall', 'square']]
    suffixes = ['stochastic'] + suffixes

    for suffix, color in zip(suffixes, colors):
        with xr.open_dataset(f'data/baseline-{suffix}.nc') as ds:
            pmf = ds['pmf_u'].values * 1000

        error = np.sqrt(((pmf - ref) ** 2).sum(axis=1))
        axes[1].plot(days, error, color=color, label=suffix.replace('-', ', '))

    for ax in axes:
        ax.set_xlim(0, 15)
        ax.set_xticks(np.linspace(0, 15, 6))
        ax.set_xlabel('time (days)')

    axes[1].set_ylim(0, 4)
    axes[1].set_ylabel('RMSE (mPa)')

    axes[1].set_title('error in momentum flux')
    axes[1].legend()

    plt.tight_layout()
    axes[1].grid(True, color='lightgray')
    plt.savefig('plots/baseline.png', dpi=400)

if __name__ == '__main__':
    config.load_config('config/baseline.toml')
    save_reference_mean_state()

    config.n_day = 15
    config.interactive_mean = False
    config.mean_file = 'data/baseline-mean.nc'
    config.refresh()

    run_refinements()
    make_summary_plot()
