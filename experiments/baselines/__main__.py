import sys

import xarray as xr

sys.path.insert(0, '.')
from msgwam import config
from msgwam.integrate import SBDF2Integrator
from msgwam.plotting import plot_integration

import strategies
from preprocess import plot_mean_state, save_mean_state
from analysis import COLORS, plot_fluxes, plot_power, plot_scores

def prepare_mean_state(mode: str) -> None:
    save_mean_state(mode)
    plot_mean_state()

def run(name: str) -> None:
    func_name = name.replace('-', '_')
    getattr(strategies, func_name)()
    config.refresh()

    average_pmf_u, average_pmf_v = 0, 0
    n_trials = 25 if name == 'stochastic' else 1

    for _ in range(n_trials):
        ds = SBDF2Integrator().integrate().to_dataset()
        average_pmf_u = average_pmf_u + ds['pmf_u'].values
        average_pmf_v = average_pmf_v + ds['pmf_v'].values

    ds['pmf_u'].values = average_pmf_u / n_trials
    ds['pmf_v'].values = average_pmf_v / n_trials

    ds.to_netcdf(f'data/baselines/{name}.nc')
    config.load_config('config/baselines.toml')

def plot(name: str) -> None:
    with xr.open_dataset(f'data/baselines/{name}.nc', use_cftime=True) as ds:
        plot_integration(ds, f'plots/baselines/{name}.png')

def analyze() -> None:
    # plot_fluxes()
    # plot_scores()
    plot_power()

if __name__ == '__main__':
    config.load_config('config/baselines.toml')

    for arg in sys.argv[1:]:
        task, *parts = arg.split(':')
        globals()[task.replace('-', '_')](*parts)
