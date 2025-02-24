from __future__ import annotations
from typing import TYPE_CHECKING

import cftime
import matplotlib.pyplot as plt
import numpy as np

from msgwam import config
from msgwam.constants import EPOCH
from msgwam.sources import get_spectrum
from msgwam.utils import cos_and_sin, get_time, open_dataset

from ..hyperparameters import scenarios as hp
from ..shared.plotting import plot_summaries
from .utils import round_sigfigs

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.lines import Line2D

_COLORS = {'u' : 'tab:red', 'v' : 'royalblue'}
_WIND_COMPONENTS = {'x' : 'u', 'y' : 'v'}

def plot_mean_state() -> None:
    """
    Plot the mean flow as a time series, and show its RMS values.
    """

    components = map(_WIND_COMPONENTS.get, hp.components)
    with open_dataset(config.prescribed_wind_file) as ds:
        datas = {c : ds[c] for c in components}
        units = ['m / s'] * len(datas)
        amaxes = [80] * len(datas)

    plot_summaries(datas, amaxes=amaxes, units=units)
    plt.savefig(f'plots/{config.name}/mean-state.png', dpi=400)

def plot_spectrum() -> None:
    """Plot the spectrum to be used by the various strategies."""

    with config.override(n_source=120):
        flux = 1000 * get_spectrum()['flux']
        amax = 0.01 * np.ceil(flux.max() / 0.01)
        cos, sin  = cos_and_sin(flux['phi'])

    widths = [4.5] * len(hp.components)
    widths = widths + ([0.2] if 'time' in flux.coords else [])
    fig, axes = plt.subplots(ncols=len(widths), width_ratios=widths)
    fig.set_size_inches(1.15 * sum(widths), 1.15 * 3)

    for c, ax in zip(hp.components, axes):
        factor = {'x' : cos, 'y' : sin}[c]
        bins, data = flux['cp'] * factor, flux * abs(factor)
        data = data.groupby(bins).sum().rename(group='cp')

        if 'time' in flux.coords:
            units = f'days since {EPOCH}'
            days = cftime.date2num(flux['time'], units)

            img = ax.pcolormesh(
                days, data['cp'],
                data.values.T,
                shading='nearest',
                vmin=0, vmax=amax,
                cmap='Reds'
            )

            cbar = plt.colorbar(img, cax=axes[-1])
            cbar.set_ticks(np.linspace(0, amax, 5))
            cbar.set_label('flux (mPa)')

            ax.set_xlim(days.min(), days.max())
            ax.set_ylim(-config.c_max, config.c_max)
            ax.set_yticks(np.linspace(*ax.get_ylim(), 5))

            ax.set_xlabel('time (days)')
            ax.set_ylabel('$c_\\mathrm{p}$ (m / s)')

        else:
            ax.scatter(data['cp'], data.values, color='k')
            ax.grid(color='lightgray')
            ax.set_axisbelow(True)

            ax.set_xlabel('$c_\\mathrm{p}$ (m / s)')
            ax.set_ylabel('flux (mPa)')

            ax.set_xlim(-config.c_max, config.c_max)
            ax.set_ylim(0, amax)

        ax.set_title(f'source $F^{c}$')

    plt.tight_layout()
    plt.savefig(f'plots/{config.name}/spectrum.png', dpi=400)

def plot_mean_scales() -> None:
    """Plot power spectra in time and height for the loaded mean wind."""

    components = map(_WIND_COMPONENTS.get, hp.components)
    with open_dataset(config.prescribed_wind_file) as ds:
        winds = {c : ds[c].values for c in components}
        z = ds['z_centers'].values / 1000

        units = f'days since {EPOCH}'
        days = cftime.date2num(ds['time'].values, units)

    fig, axes = plt.subplots(nrows=2)
    fig.set_size_inches(4.5, 6)

    dt = days[1] - days[0]
    dz = z[1] - z[0]

    lines = []
    for c, data in winds.items():
        line = _plot_power_spectrum(axes[0], data.T, dt, 'days', _COLORS[c])
        _ = _plot_power_spectrum(axes[1], data, dz, 'km', _COLORS[c])
        lines.append(line)

    if len(lines) > 1:
        handles = [f'$\\bar{{{c}}}$' for c in winds]
        axes[0].legend(lines, handles) 

    plt.tight_layout()
    plt.savefig(f'plots/{config.name}/mean-scales.png', dpi=400)

def _plot_power_spectrum(
    ax: Axes,
    a: np.ndarray,
    d: float,
    xlabel: str, 
    color: str
) -> Line2D:
    """
    Plot a power spectrum on the given axis.

    Parameters
    ----------
    ax
        Axis on which to plot the spectrum.
    a
        Data to Fourier-transform. If there is more than one dimension, the FFT
        will be applied over the last one and the power of each frequency will
        be averaged over all samples.
    d
        Sample spacing, in whatever units should be used.
    xlabel
        String to use to label the frequency axis.
    color
        Color in which to plot the spectrum.

    Returns
    -------
    Line2D
        Line describing the power spectrum.

    """

    power = (abs(np.fft.rfft(a)[:, 1:]) ** 2).mean(0)
    freqs = np.fft.rfftfreq(a.shape[1], d)[1:]
    lfreqs = np.log(freqs)

    ticks = np.exp(np.linspace(lfreqs.min(), lfreqs.max(), 10))
    labels = round_sigfigs(1 / ticks, 2)
    ticks = 1 / labels

    line, = ax.plot(freqs, power, color=color)
    ax.grid(color='lightgray')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.minorticks_off()

    cmin, cmax = ax.get_ylim()
    amin = 10 ** np.floor(np.log10(power.min()))
    amax = 10 ** np.ceil(np.log10(power.max()))

    ax.set_xticks(ticks, labels, rotation=45)
    ax.set_ylim(min(amin, cmin), max(amax, cmax))

    ax.set_xlabel(xlabel)
    ax.set_ylabel('power')

    return line
