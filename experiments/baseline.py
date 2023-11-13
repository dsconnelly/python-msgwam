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
    dr_init = config.dr_init * 4
    dm_init = config.dm_init * 4
    n_per_mode = config.n_per_mode // 4
    n_ray_max = config.n_ray_max // 16

    for i in range(1, 5):
        config.dr_init = dr_init / i
        config.dm_init = dm_init / i
        config.n_per_mode = n_per_mode * i
        config.n_ray_max = n_ray_max * (i ** 2)

        config.refresh()
        ds = integrate.SBDF2Integrator().integrate().to_dataset()
        ds.to_netcdf(f'data/baseline-{i ** 2}x.nc')

    config.n_ray_max = n_ray_max
    config.refresh()

    ds = integrate.SBDF2Integrator().integrate().to_dataset()
    ds.to_netcdf(f'data/baseline-1x-small.nc')

def plot_errors():
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 6)

    with xr.open_dataset('data/baseline-16x.nc') as ds:
        days = ds['time'].values / (60 * 60 * 24)
        ref = ds['pmf_u'].values * 1000

    colors = ['tab:red', 'royalblue', 'forestgreen']
    for i, color  in enumerate(colors, start=1):
        with xr.open_dataset(f'data/baseline-{i ** 2}x.nc') as ds:
            pmf = ds['pmf_u'].values * 1000
            
        error = np.sqrt(((pmf - ref) ** 2).sum(axis=1))
        ax.plot(days, error, color=color, label=f'{i ** 2}x')

    with xr.open_dataset(f'data/baseline-1x-small.nc') as ds:
        pmf = ds['pmf_u'].values * 1000
        
    error = np.sqrt(((pmf - ref) ** 2).sum(axis=1))
    ax.plot(days, error, color='darkviolet', label=f'1x (small)')

    ax.set_xlim(0, 15)
    ax.set_ylim(0, 12)
    ax.set_xticks(np.arange(16, step=3))

    ax.set_xlabel('time (days)')
    ax.set_ylabel('RMSE in MF (mPa)')
    ax.legend()

    plt.tight_layout()
    ax.grid(True, color='lightgray')
    plt.savefig('plots/baseline.png', dpi=400)

if __name__ == '__main__':
    config.load_config('config/baseline.toml')
    BASELINE = vars(config).copy()
    # save_reference_mean_state()

    config.n_day = 15
    config.interactive_mean = False
    config.mean_file = 'data/baseline-mean.nc'

    run_refinements()
    plot_errors()
