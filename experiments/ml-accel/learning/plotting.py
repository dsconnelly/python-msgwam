from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from msgwam import config
from msgwam.dispersion import get_cp_x
from msgwam.utils import get_vertical_grids

from . import hyperparameters as hp
from .utils import get_indices, get_overrides, load_data

_COLORS = {
    'fine' : 'forestgreen',
    'coarse' : 'royalblue'
}

_MODEL_COLORS = [
    'fuchsia',
    'darkviolet',
    'tab:red'
]

def plot_training_samples(*args: str) -> None:
    """
    
    """

    n_rows, n_cols = 2, 4
    fig, axes = plt.subplots(n_rows, n_cols)
    fig.set_size_inches(n_cols * 3, n_rows * 4.5)
    axes = axes.flatten()

    idx = get_indices('validation')[0]
    n_samples = min(len(axes), len(idx))
    idx = np.random.choice(idx, n_samples, replace=False)

    u, rays, _ = load_data('flux', 'coarse')
    u, rays = u[idx], rays[idx]

    datas, colors, labels = [], [], []
    with config.override(n_grid=get_overrides()['n_grid']):
        z_faces, z_centers = [z / 1e3 for z in get_vertical_grids()]

    for arg in args:
        if arg.startswith('flux'):
            _, grain = arg.split('-')
            data = load_data('flux', grain)[-1][idx]
            color, label = _COLORS[grain], arg

        elif arg.startswith('coeffs'):
            _, grain = arg.split('-')
            data = load_data('coeffs', grain, reconstructed=True)[-1][idx]
            color, label = _MODEL_COLORS.pop(), arg

        elif arg.endswith('.jit'):
            data = torch.jit.load(arg)(u, rays)
            k = arg.split('-')[-1].split('.')[0]
            color, label = _MODEL_COLORS.pop(), f'network {k}'

        datas.append(data)
        colors.append(color)
        labels.append(label)

    for n, (i, ax) in enumerate(zip(idx, axes)):
        handles = []

        for data, color in zip(datas, colors):
            line, = ax.plot(data[n], z_faces, color=color)
            handles.append(line)

        ax.set_xlim(-1.25, 1.25)
        ax.set_ylim(z_faces.min(), z_faces.max())

        ax.set_title(f'sample {i}')
        ax.set_xlabel('normalized flux')
        ax.set_ylabel('height (km)')

        ax.grid(color='lightgray')
        ax.tick_params('both', direction='in')

        ax = ax.twiny()
        line, = ax.plot(u[n], z_centers, color='k')
        handles.append(line)

        k, l, m, *_ = rays[n]
        ones = np.ones_like(z_centers)
        cp_x = get_cp_x(k, l, m, config.N_ref) * ones + u[n, 0]
        line, = ax.plot(cp_x, z_centers, color='gray', ls='dashed')
        handles.append(line)

        ax.set_xlim(-50, 50)
        ax.set_xlabel('$\\bar{u}$ (m / s)')

        if n == 0:
            side = 'left' if k > 0 else 'right'
            labels = labels + ['$\\bar{u}$', '$c_{\\mathrm{p}}$']
            ax.legend(handles, labels, loc=f'lower {side}')

    plt.tight_layout()
    plt.savefig(f'plots/{config.name}/training-samples.png', dpi=400)
