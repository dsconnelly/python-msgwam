import tomllib

import matplotlib.pyplot as plt
import numpy as np
import torch

from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

from msgwam import config
from msgwam.dispersion import get_cp_x, get_omega_hat
from msgwam.utils import get_vertical_grids

from .. import hyperparameters as hp
from .utils import get_overrides, load_data

_COLORS = {
    'fine' : 'forestgreen',
    'coarse' : 'royalblue'
}

_MODEL_COLORS = [
    'fuchsia',
    'darkviolet',
    'tab:red'
]

def plot_cv_scores(target_type: str) -> None:
    """
    Plot the average cross-validation scores for each hyperparameters.

    Parameters
    ----------
    target_type
        String identifying the target type, as passed to `train_network`.

    """

    with open(hp.grid_path, 'rb') as f:
        options, _ = hp._parse_grid(tomllib.load(f))

    mesh = np.meshgrid(*options.values(), indexing='ij')
    params = np.stack(mesh, axis=0).reshape(len(options), -1)
    means = {name : np.zeros(len(v)) for name, v in options.items()}

    log_dir = f'logs/{config.name}'
    _, grain = target_type.split('-')
    best_score, best_k = np.inf, None

    for k in range(params.shape[1]):
        with open(f'{log_dir}/train-surrogate-{grain}-{k}.out') as f:
            line = [s for s in f.readlines() if s.startswith('Best')][0]
            score = float(line.strip().split()[-1])

            if score < best_score:
                best_score = score
                best_k = k

        for i, (name, values) in enumerate(options.items()):
            j = values.index(params[i, k])
            means[name][j] += score

    fig, ax = plt.subplots()
    fig.set_size_inches(5.5, 4.5)
    ax.invert_yaxis()

    means = {name : v * len(v) / params.shape[1] for name, v in means.items()}
    amax = max(sum([v.tolist() for v in means.values()], []))
    norm = Normalize(0,0.01 * np.ceil(amax / 0.01))
    cmap = plt.cm.get_cmap('Reds')

    for i, (name, data) in enumerate(means.items()):
        colors = cmap(norm(data))
        width = 1 / len(data)

        for j, color in enumerate(colors):
            ax.add_patch(Rectangle(
                (j * width, i - 0.5),
                width=width, height=1,
                ec='none', fc=color
            ))

            value = options[name][j]
            ax.text(
                (j + 0.5) * width, i,
                s='$\\bf{' + f'{value}:' + '}$' + f' {data[j]:.4f}',
                size='large',
                ha='center',
                va='center'
            )

        j = options[name].index(params[i, best_k])
        ax.add_patch(Rectangle(
            (j * width, i - 0.5),
            width=width, height=1,
            ec='k', fc='none',
            linewidth=1.5,
            clip_on=False,
            zorder=10
        ))

    ax.spines[['left', 'right', 'top', 'bottom']].set_visible(False)
    ax.tick_params('both', color=[0, 0, 0, 0])

    ax.set_ylim(len(means) - 0.5, -0.5)
    ax.set_yticks(np.arange(len(means)))
    ax.set_yticklabels([k.split('.')[-1] for k in means.keys()])
    ax.set_xticks([])

    plt.tight_layout()
    plt.savefig(f'plots/{config.name}/{target_type}-cv.png', dpi=400)

def plot_network_errors(model_path: str) -> None:
    """
    Plot some diagnostics of neural network errors. The left panel shows RMSEs
    as a function of height, while the right panel decomposes RMSE over zonal
    phase speed as a histogram.

    Parameters
    ----------
    model_path
        Path where JITted neural network is saved.

    """

    widths = [3, 1.5 * 4.5]
    fig, axes = plt.subplots(ncols=2, width_ratios=widths)
    fig.set_size_inches(sum(widths), 4.5)

    idx_tr, idx_ev = get_indices('validation')
    idx_tr = np.random.choice(idx_tr, min(50000, len(idx_tr)), replace=False)
    idx_ev = np.random.choice(idx_ev, min(50000, len(idx_ev)), replace=False)

    colors = ['forestgreen', 'tab:red']
    labels = ['training', 'validation']

    with config.override(n_grid=get_overrides()['n_grid']):
        z = get_vertical_grids()[0] / 1e3

    u, rays, targets = load_data('flux-coarse')
    model = torch.jit.load(model_path)
    
    for idx, color, label in zip([idx_tr, idx_ev], colors, labels):
        error = targets[idx] - model(u[idx], rays[idx])
        rmse = torch.sqrt((error ** 2).mean(dim=0))
        axes[0].plot(rmse, z, color=color, label=label)

    axes[0].set_xlim(0, .2)
    axes[0].set_ylim(z.min(), z.max())

    axes[0].legend(loc='lower right')
    axes[0].set_xlabel('normalized RMSE')
    axes[0].set_ylabel('height (km)')

    axes[0].grid(color='lightgray')
    axes[0].tick_params('both', direction='in')

    u, rays = u[idx_tr], rays[idx_tr]
    k, l, m, dk, dl, dm, dens = rays.T
    omega_hat = get_omega_hat(k, l, m, config.N_ref)
    cp_x = omega_hat / k + u[:, 0, 0]

    targets = targets[idx_tr]
    error = targets - model(u, rays)
    rmse = torch.sqrt((error ** 2).mean(dim=1))

    coord = cp_x
    edges = np.linspace(coord.min(), coord.max(), 21)
    h, _ = np.histogram(coord, bins=edges, weights=rmse)
    count, _ = np.histogram(coord, bins=edges)

    width = edges[1] - edges[0]
    x = (edges[:-1] + edges[1:]) / 2
    axes[1].bar(x, h / count, width=width, fc='lightgray', ec='k')
    axes[1].twinx().plot(x, count, color='k')

    axes[1].set_xlim(edges[0], edges[-1])
    axes[1].set_ylim(0, 0.2)

    plt.tight_layout()
    plt.savefig(f'plots/{config.name}/network-errors.png', dpi=400)

def plot_training_samples(*args: str) -> None:
    """
    Plot training and evaluation samples from various sources.

    Parameters
    ----------
    args
        List of profile sources to plot. Can either be a valid `target_type` to
        pass to `load_data`, or the path to a JITted trained `Surrogate`.

    """

    n_rows, n_cols = 2, 4
    fig, axes = plt.subplots(n_rows, n_cols)
    fig.set_size_inches(n_cols * 3, n_rows * 4.5)
    axes = axes.flatten()

    idx_tr, idx_ev = get_indices('validation')
    idx_tr = np.random.choice(idx_tr, n_cols, replace=False)
    idx_ev = np.random.choice(idx_ev, n_cols, replace=False)
    idx = np.concatenate((idx_tr, idx_ev))

    u, rays, _ = load_data('flux-coarse')
    u, rays = u[idx], rays[idx]

    datas, colors, labels = [], [], []
    with config.override(n_grid=get_overrides()['n_grid']):
        z_faces, z_centers = [z / 1e3 for z in get_vertical_grids()]

    for arg in args:
        if arg.startswith('flux'):
            _, grain = arg.split('-')
            data = load_data(arg)[-1][idx]
            color, label = _COLORS[grain], arg

        elif arg.startswith('proxies'):
            data = load_data(arg, reconstructed=True)[-1][idx]
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

        suffix = 'train' if n < n_cols else 'eval'
        ax.set_title(f'sample {i} ({suffix})')
        ax.set_xlabel('normalized flux')
        ax.set_ylabel('height (km)')

        ax.grid(color='lightgray')
        ax.tick_params('both', direction='in')

        ax = ax.twiny()
        line, = ax.plot(u[n, 0], z_centers, color='k')
        handles.append(line)

        k, l, m, *_ = rays[n]
        ones = np.ones_like(z_centers)
        cp_x = get_cp_x(k, l, m, config.N_ref) * ones + u[n, 0, 0]
        line, = ax.plot(cp_x, z_centers, color='gray', ls='dashed')
        handles.append(line)

        ax.set_xlim(-60, 60)
        ax.set_xlabel('$\\bar{u}$ (m / s)')

        if n == 0:
            side = 'left' if k > 0 else 'right'
            labels = labels + ['$\\bar{u}$', '$c_{\\mathrm{p}}$']
            ax.legend(handles, labels, loc=f'lower {side}')

    plt.tight_layout()
    plt.savefig(f'plots/{config.name}/training-samples.png', dpi=400)
