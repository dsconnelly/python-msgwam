import matplotlib.pyplot as plt

from msgwam import config

from ..shared.plotting import plot_summaries

from .utils import load_data

def plot_strategy(strategy: str) -> None:
    """
    Plot a summary of the integration outputs, including the momentum flux and
    acceleration time series as well as the RMS profiles of each quantity.

    Parameters
    ----------
    strategy
        Configuration strategy for which to plot the integration output.

    """

    zipped = zip(['flux', 'acceleration'], [1e3, 86400])
    datas = {s : f * load_data(strategy, s, 0) for s, f in zipped}
    plot_summaries(datas, amaxes=[3, 40], units=['mPa', 'm / s / day'])
    plt.savefig(f'plots/{config.name}/{strategy}.png', dpi=400)