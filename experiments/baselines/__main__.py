import sys

import xarray as xr

sys.path.insert(0, '.')
from msgwam import config
from msgwam.integrate import SBDF2Integrator
from msgwam.plotting import plot_integration

import strategies
from jra import plot_mean_state, save_mean_state

def prepare_mean_state() -> None:
    save_mean_state()
    plot_mean_state()

def run(name: str) -> None:
    func_name = name.replace('-', '_')
    getattr(strategies, func_name)()
    config.refresh()

    solver = SBDF2Integrator().integrate()
    solver.to_dataset().to_netcdf(f'data/baselines/{name}.nc')
    config.load_config('config/baselines.toml')

def plot(name: str) -> None:
    with xr.open_dataset(f'data/baselines/{name}.nc', use_cftime=True) as ds:
        plot_integration(ds, f'plots/baselines/{name}.png')

def plot_count() -> None:
    with xr.open_dataset(f'data/baselines/reference.nc', use_cftime=True) as ds:
        import cftime, numpy as np
        from msgwam.constants import EPOCH
        alive = (~np.isnan(ds['meta'].values)).sum(axis=1)
        time = cftime.date2num(ds['time'].values, units=f'days since {EPOCH}')

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    fig.set_size_inches(6.5, 4)

    ax.plot(time, alive, color='k')
    ax.set_xlabel('time (days)')
    ax.set_ylabel('ray volumes')

    plt.tight_layout()
    plt.savefig('plots/baselines/count.png', dpi=400)

if __name__ == '__main__':
    config.load_config('config/baselines.toml')

    for arg in sys.argv[1:]:
        task, *parts = arg.split(':')
        globals()[task.replace('-', '_')](*parts)
