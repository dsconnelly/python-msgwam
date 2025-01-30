import matplotlib.pyplot as plt

from msgwam import config
from msgwam.utils import open_dataset

from ..shared.plotting import plot_summaries

def plot_mean_state() -> None:
    """
    
    """

    with open_dataset(config.prescribed_wind_file) as ds:
        datas = {'u' : ds['u']}

    plot_summaries(datas, amaxes=[60], units=['m / s'])
    plt.savefig(f'plots/{config.name}/mean-state.png', dpi=400)
