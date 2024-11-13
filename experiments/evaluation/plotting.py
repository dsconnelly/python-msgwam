from __future__ import annotations
from typing import TYPE_CHECKING, Iterable, Optional
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from matplotlib.colors import LinearSegmentedColormap

from msgwam import config

if TYPE_CHECKING:
    from matplotlib.axes import Axes

_COLORS = [
    'forestgreen',
    'royalblue',
    'tab:red',
    'darkviolet',
    'goldenrod'
]

def plot_error_grid(
    drs: list[int],
    n_sources: list[int],
    errors: np.ndarray,
    axes: Optional[tuple[Axes, Axes]]=None
) -> tuple[Axes, Axes]:
    """
    Plot normalized errors for a grid of resolution settings.

    Parameters
    ----------
    drs
        Array of `config.dr_init` values corresponding to rows of `errors`.
    n_sources
        Array of `config.n_source` values corresponding to columns of `errors`.
    errors
        Array of normalized errors.
    axes
        Axes on which to plot the grid and the colorbar. If `None`, two new axes
        will be created to hold the plot.

    Returns
    -------
    tuple[Axes, Axes]
        Axes containing the plot and the colorbar. If these were passed as
        arguments, the same objects are returned.

    """

    if axes is None:
        widths = [4.5, 0.2]
        fig, axes = plt.subplots(ncols=2, width_ratios=widths)
        fig.set_size_inches(sum(widths), 4.5)

    colors = ['darkgreen', 'w', 'darkred']
    cmap = LinearSegmentedColormap.from_list('custom', colors, 256)

    ax, cax = axes
    img = ax.imshow(
        errors.T,
        vmin=0, vmax=2,
        origin='lower',
        aspect='auto',
        cmap=cmap
    )

    dcs = [round(_get_dc(n), 2) for n in n_sources]
    ax.set_xticks(np.arange(len(drs)), labels=drs, rotation=45)
    ax.set_yticks(np.arange(len(n_sources)), labels=dcs)

    ax.set_xlabel('$\\delta z$ (m)')
    ax.set_ylabel('$\\delta c_{\mathrm{p}}$ (m / s)')

    cbar = plt.colorbar(img, cax=cax)
    cbar.set_ticks(np.linspace(0, 2, 5))
    cbar.set_label('normalized error')

    return ax, cax

def plot_rmse_profiles(
    z: np.ndarray,
    profiles: list[np.ndarray],
    colors: Iterable[str]=cycle(_COLORS),
    linestyles: Iterable[str]=cycle(['solid']),
    labels: Iterable[str]=cycle(['']),
    ax: Optional[Axes]=None
) -> Axes:
    """
    Plot root-mean-square errors in momentum flux as a function of height.

    Parameters
    ----------
    z
        Height coordinate to plot with.
    profiles
        Precomputed error profiles to plot.
    colors
        Colors to use for each profile. If not provided, a reasonable list of
        colors will be cycled through.
    linestyles
        Style to use for each profile. If not provided, all lines will be solid.
    labels
        Label for each profile. If not provided, no legend will be created.
    ax
        Axis on which to plot. If `None`, a new axis will be created.

    Returns
    -------
    Axes
        Axis with the plot. If `ax` was passed, the same object is returned.

    """

    if ax is None:
        fig, ax = plt.subplots()
        fig.set_size_inches(3, 4.5)

    for profile, color, ls, label in zip(profiles, colors, linestyles, labels):
        ax.plot(1000 * profile, z, color=color, ls=ls, label=label)

    _, x_max = ax.get_xlim()
    x_max = np.ceil(x_max / 0.5) * 0.5
    y_min = config.z_min / 1000
    y_max = config.z_max / 1000

    ax.set_xlim(0, x_max)
    ax.set_ylim(y_min, y_max)
    ax.grid(color='lightgray')
    
    ax.set_xlabel('RMSE (mPa)')
    ax.set_ylabel('height (km)')
    ax.tick_params('both', direction='in')

    return ax

def _get_dc(n: int) -> float:
    """
    Get the phase velocity resolution corresponding to a particular number of
    spectral elements at the source.

    Parameters
    ----------
    n
        Number of spectral source elements

    Returns
    -------
    float
        Extent of each source ray volume in phase velocity space.

    """

    return np.diff(np.linspace(-config.c_max, config.c_max, n + 1))[0]
