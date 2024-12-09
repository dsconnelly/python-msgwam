from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from msgwam import config
from msgwam.dispersion import get_cp_x
from msgwam.utils import get_vertical_grids

from .utils import get_indices, load_data

def plot_training_samples(model_path: Optional[str]=None) -> None:
    """
    
    """

    idx, _ = get_indices('validation')
    u, X, Y_coarse = load_data('coarse')
    *_, Y_fine = load_data('fine')
    u, X = u[idx], X[idx]

    datas = [Y_fine[idx], Y_coarse[idx]]
    colors = ['forestgreen', 'royalblue']
    labels = ['fine', 'coarse']

    if model_path is not None:
        model = torch.jit.load(model_path)
    
        with torch.no_grad():
            datas.append(model(u, X))

        colors.append('tab:red')
        labels.append('network')

    n_rows, n_cols = 2, 4
    fig, axes = plt.subplots(n_rows, n_cols)
    fig.set_size_inches(n_cols * 3, n_rows * 4.5)
    axes = axes.flatten()

    n_samples = min(len(axes), u.shape[0])
    jdx = np.random.choice(u.shape[0], size=n_samples, replace=False)
    z_faces, z_centers = [z / 1e3 for z in get_vertical_grids()]

    for i, (j, ax) in enumerate(zip(jdx, axes)):
        handles = []
        for data, color in zip(datas, colors):
            handles.append(ax.plot(data[j], z_faces, color=color)[0])

        ax.set_xlim(-1.25, 1.25)
        ax.set_ylim(z_faces.min(), z_faces.max())

        ax.set_xlabel(f'normalized flux')
        ax.set_ylabel('height (km)')

        ax.grid(color='lightgray')
        ax.tick_params('both', direction='in')

        ax = ax.twiny()
        line, = ax.plot(u[j], z_centers, color='k')
        handles.append(line)

        k, l, m, *_ = X[j]
        ones = np.ones_like(z_centers)
        cp_x = get_cp_x(k, l, m, config.N_ref) * ones + u[j, 0]
        handles.append(ax.plot(cp_x, z_centers, color='gray', ls='dashed')[0])

        if i == 0:
            labels = labels + ['$\\bar{u}$', '$c_{\\mathrm{p}}$']
            ax.legend(handles, labels)

        ax.set_xlim(-50, 50)
        ax.set_xlabel('$\\bar{u}$ (m / s)')

    plt.tight_layout()
    plt.savefig(f'plots/{config.name}/training-samples.png', dpi=400)