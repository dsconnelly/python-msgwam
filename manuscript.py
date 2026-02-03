import sys

import cftime
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, '.')
from msgwam import config
from msgwam.sources import get_spectrum
from msgwam.utils import get_time, get_vertical_grids, open_dataset

# from acceleration.shared.filtering import gaussian_filter

def plot_spectra():
    """
    
    """

    fig, axes = plt.subplots(ncols=2)
    fig.set_size_inches(2 * 4.5, 3)

    for j in range(2):
        with config.override(equal_flux=(j == 1)):
            ds = get_spectrum()
            ds = ds.isel(channel=(ds['phi'] == 0))

            c_mid = ds['cp']
            edges = np.zeros(len(c_mid) + 1)

            for i, c in enumerate(c_mid, start=1):
                dc = 2 * (c - edges[i - 1])
                edges[i] = edges[i - 1] + dc

            widths = edges[1:] - edges[:-1]
            flux = 1000 * ds['flux'].values
            x = c_mid.values

            kwargs = dict(ec='k', fc='lightgray')
            axes[j].bar(x, flux, width=widths, **kwargs)
            axes[j].bar(-x, flux, width=widths, **kwargs)

        axes[j].set_xlim(-50, 50)
        axes[j].set_ylim(0, 1.5)

        axes[j].set_xticks(np.linspace(-50, 50, 5))
        axes[j].set_yticks(np.linspace(0, 1.5, 4))

        axes[j].set_title(f'({"a" if j == 0 else "b"})')

        axes[j].set_xlabel('phase speed (m s$^{-1}$)')
        
        if j == 0:
           axes[j].set_ylabel('source flux (mPa)')

    plt.tight_layout()
    plt.savefig('plots/manuscript/spectra.png', dpi=400, bbox_inches='tight')

def plot_example_scenarios(mode: str='winds') -> None:
    """Plot mean wind time series for several MiMA scenarios."""

    widths = [4.5, 4.5, 0.2]
    fig, axes = plt.subplots(3, len(widths), width_ratios=widths)
    axes, caxes = axes[:, :-1], axes[:, -1]
    fig.set_size_inches(sum(widths), 9)
    
    if mode == 'winds':
        for i in (0, 2):
            caxes[i].set_axis_off()

    days = np.linspace(0, config.n_day, config.n_steps)
    z = get_vertical_grids()[['fluxes', 'winds'].index(mode)] / 1000
    amaxes = {'winds' : [100] * 3, 'fluxes' : [10, 6, 3]}[mode]
    names = ['lisbon', 'miami', 'maldives']

    for i, (name, amax) in enumerate(zip(names, amaxes)):
        for j, (data, label) in enumerate(_iterate_datas(name, mode)):
            img = axes[i, j].pcolormesh(
                days, z, data.values.T,
                vmin=-amax, vmax=amax,
                shading='nearest',
                cmap='RdBu_r'
            )

            axes[i, j].set_xlim(0, config.n_day)
            axes[i, j].set_ylim(10, 60)

            xticks = np.linspace(0, config.n_day, 6)
            axes[i, j].set_xticks(xticks.astype(int))

            yticks = np.linspace(10, 60, 6)
            axes[i, j].set_yticks(yticks.astype(int))

            if i == 2:
                axes[i, j].set_xlabel('days')
            else:
                axes[i, j].set_xticklabels([])

            if j == 0:
                axes[i, j].set_ylabel('height (km)')
            else:
                axes[i, j].set_yticklabels([])

            letter = chr(2 * i + j + 97)
            display = _format_name(name)
            axes[i, j].set_title(f'({letter}) {display} ${label}$')

        if (i == 1) or (mode == 'fluxes'):
            cbar = plt.colorbar(img, cax=caxes[i])
            cbar.set_ticks(np.linspace(-amax, amax, 5))
            cbar.set_label('m / s' if mode == 'winds' else 'mPa')

    plt.tight_layout()
    plt.savefig(
        f'plots/manuscript/example-{mode}.png',
        bbox_inches='tight',
        dpi=400
    )

def _format_name(name: str) -> str:
    """Format a scenario name for display."""

    return ' '.join(map(lambda s: s.capitalize(), name.split('-')))

def _iterate_datas(name: str, mode: str):
    """
    
    """

    if mode == 'winds':
        path = f'data/mima-{name}/input/mean-state.nc'
        with open_dataset(path).interp(time=get_time()) as ds:
            for c in 'uv': yield ds[c], f'\\bar{{{c}}}'

    elif mode == 'fluxes':
        path = f'data/mima-{name}/strategies/reference.nc'
        with open_dataset(path) as ds:
            for c in 'xy':
                data = 0
                for suffix in {'x' : 'ew', 'y' : 'ns'}[c]:
                    data = data + ds[f'pmf_{suffix}']

                data = gaussian_filter(data, hours=3, z_faces=1000)
                yield 1000 * data, f'F_{{{c}}}'

if __name__ == '__main__':
    config.load('config/mima-amundsen-sea.toml')
    # plot_example_scenarios('winds')
    # plot_example_scenarios('fluxes')

    plot_spectra()