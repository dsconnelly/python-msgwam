import cftime
import matplotlib.gridspec as gs
import matplotlib.pyplot as plt
import numpy as np

from msgwam import config
from msgwam.constants import EPOCH
from msgwam.utils import open_dataset

from ..shared.plotting import plot_summaries

def plot_mean_state() -> None:
    """
    Plot the mean flow as a time series, and show its RMS values.
    """

    with open_dataset(config.prescribed_wind_file) as ds:
        datas = {'u' : ds['u']}

    plot_summaries(datas, amaxes=[60], units=['m / s'])
    plt.savefig(f'plots/{config.name}/mean-state.png', dpi=400)

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