import cftime
import matplotlib.pyplot as plt
import numpy as np

from msgwam import config
from msgwam.constants import EPOCH
from msgwam.sources import get_spectrum
from msgwam.utils import open_dataset

from ..shared.plotting import plot_summaries

def plot_mean_state() -> None:
    """
    Plot the mean flow as a time series, and show its RMS values.
    """

    with open_dataset(config.prescribed_wind_file) as ds:
        datas = {'u' : ds['u'], 'v' : ds['v']}

    plot_summaries(datas, amaxes=[60, 60], units=['m / s', 'm / s'])
    plt.savefig(f'plots/{config.name}/mean-state.png', dpi=400)

def plot_spectrum() -> None:
    """Plot the spectrum to be used by the various strategies."""

    with config.override(n_source=120, dt_launch=config.dt):
        flux = 1000 * get_spectrum()['flux']

    widths = [4.5, 0.2]
    fig, (ax, cax) = plt.subplots(ncols=2, width_ratios=widths)
    fig.set_size_inches(1.15 * sum(widths), 1.15 * 3)

    units = f'days since {EPOCH}'
    days = cftime.date2num(flux['time'], units)
    amax = 0.01 * np.ceil(flux.max() / 0.01)
    
    img = ax.pcolormesh(
        days, flux['cp_x'], flux.values.T,
        shading='nearest',
        cmap='Reds',
        vmin=0,
        vmax=amax
    )

    cbar = plt.colorbar(img, cax=cax)
    cbar.set_label('source flux (mPa)')
    cbar.set_ticks(np.linspace(0, amax, 5))

    ax.set_xlim(days.min(), days.max())
    ax.set_ylim(-config.c_max, config.c_max)
    ax.set_yticks(np.linspace(*ax.get_ylim(), 5))

    ax.set_xlabel('time (days)')
    ax.set_ylabel('$c_\\mathrm{p}$ (m / s)')

    plt.tight_layout()
    plt.savefig(f'plots/{config.name}/spectrum.png', dpi=400)

def plot_windows() -> None:
    """
    Plot the horizontal phase speeds that are blocked or transmitted by the mean
    flow as a function of time.
    """

    fig, ax = plt.subplots()
    fig.set_size_inches(4.5, 3)

    with open_dataset(config.prescribed_wind_file) as ds:    
        days = cftime.date2num(ds['time'], f'days since {EPOCH}')
        u = ds['u'].sel(z_centers=slice(30e3, 60e3))
    
    u_min, u_max = u.min('z_centers'), u.max('z_centers')
    ax.fill_between(days, u_min, u_max, color='lightgray')

    ax.set_xlim(days.min(), days.max())
    ax.set_ylim(-config.c_max, config.c_max)
    ax.set_yticks(np.linspace(-config.c_max, config.c_max, 5))

    ax.set_xlabel('time (days)')
    ax.set_ylabel('phase velocity (m / s)')

    plt.tight_layout()
    plt.savefig(f'plots/{config.name}/windows.png', dpi=400)