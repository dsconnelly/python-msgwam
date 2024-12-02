import matplotlib.pyplot as plt
import numpy as np

from msgwam import config
from msgwam.dispersion import get_cp_x
from msgwam.utils import get_vertical_grids

from . import hyperparameters as hp

def plot_training_samples() -> None:
    """
    
    """

    u = np.load(f'data/{config.name}/u.npy')
    X = np.load(f'data/{config.name}/X.npy')
    Y_fine = np.load(f'data/{config.name}/Y-fine.npy')
    Y_coarse = np.load(f'data/{config.name}/Y-coarse.npy')

    n_rows, n_cols = 2, 4
    fig, axes = plt.subplots(n_rows, n_cols)
    fig.set_size_inches(n_cols * 3, n_rows * 4.5)
    axes = axes.flatten()

    jdx = np.random.choice(u.shape[0], size=len(axes), replace=False)
    z_faces, z_centers = [z / 1e3 for z in get_vertical_grids()]
    T = hp.max_days * 86400

    for i, (j, ax) in enumerate(zip(jdx, axes)):
        k, l, m, dk, dl, dm, dens = X[j]
        action = dens * dk * dl * dm

        factor = abs(k) * action * config.dr_init / T
        colors = ['forestgreen', 'royalblue']

        handles = []
        for data, color in zip([Y_fine, Y_coarse], colors):
            line, = ax.plot(data[j] / factor, z_faces, color=color)
            handles.append(line)

        ax.set_xlim(-1.25, 1.25)
        ax.set_ylim(z_faces.min(), z_faces.max())

        ax.set_xlabel(f'flux ($\\times${1e6 * factor:.2f} $\\mu$Pa)')
        ax.set_ylabel('height (km)')

        ax.grid(color='lightgray')
        ax.tick_params('both', direction='in')

        ax = ax.twiny()
        line, = ax.plot(u[j], z_centers, color='k')
        handles.append(line)

        cp_x = get_cp_x(k, l, m, config.N_ref) * np.ones_like(z_centers)
        line, = ax.plot(cp_x + u[j, 0], z_centers, color='gray', ls='dashed')
        handles.append(line)

        if i == 0:
            labels = ['fine', 'coarse', '$\\bar{u}$', '$c_{\\mathrm{p}}$']
            ax.legend(handles, labels)

        ax.set_xlim(-50, 50)
        ax.set_xlabel('$\\bar{u}$ (m / s)')

    plt.tight_layout()
    plt.savefig(f'plots/{config.name}/training-samples.png', dpi=400)