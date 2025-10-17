import json

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr

from matplotlib.colors import LinearSegmentedColormap as LSC

from msgwam.utils import get_vertical_grids

from .training import BulkLoss, iter_paths, parse_integrations, prepare_data

def plot_training_errors(
    n_bins_str: str,
    model_path: str
) -> None:
    """
    Plot RMS errors in output of a trained neural network.

    Parameters
    ----------
    n_bins_str
        How many phase speed bins the network uses.
    model_path
        Path to the JITted model.

    """

    n_bins = int(n_bins_str)
    C, M, Y = parse_integrations(cached=True)
    (C, M, Y, W), idxs, _ = prepare_data(
        n_bins,
        eval_type='te',
        arrays=(C, M, Y),
        n_samples=300000,
        transform_inputs=False
    )

    tensors = [torch.as_tensor(a)[idxs[0]] for a in (Y, W)]
    loss_func = BulkLoss(*tensors)

    C, M = torch.as_tensor(C), torch.as_tensor(M)
    Y_hat, D_hat = torch.zeros_like(M), torch.zeros_like(M[:, 0])
    model = torch.jit.load(model_path)
    batch_size, i = 4096, 0

    while i * batch_size < M.shape[0]:
        start, end = i * batch_size, min(M.shape[0], (i + 1) * batch_size)
        Y_hat[start:end], D_hat[start:end] = model(C[start:end], M[start:end])
        i = i + 1

    Y_hat = torch.cat((Y_hat, D_hat[:, None]), dim=1).numpy()
    W_hat = Y_hat.sum(axis=2, keepdims=True)
    kdx = (W_hat > 0)[..., 0]
    Y_hat[kdx] /= W_hat[kdx]

    fig, axes = plt.subplots(ncols=(n_bins + 1))
    fig.set_size_inches(3 * (n_bins + 1), 4.5)
    z = get_vertical_grids()[1] / 1000

    scales_Y = loss_func._scales_Y.numpy()
    scales_W = loss_func._scales_W.numpy()

    for j, ax, in enumerate(axes):
        rmses_W = []
        for i, idx in enumerate(idxs):
            label = ['training', 'test'][i]
            color = ['forestgreen', 'tab:red'][i]
            
            error = (Y[idx, j] - Y_hat[idx, j]) / scales_Y[j]
            loss = np.sqrt((error ** 2).mean(axis=0))
            ax.plot(loss, z, color=color, label=label)

            error = (W[idx, j] - W_hat[idx, j]) / scales_W[j]
            rmses_W.append(np.sqrt((error[..., 0] ** 2).mean(axis=0)))

        # rms = loss_func._scales_Y[j]
        # ax.plot(rms, z, color='gray', ls='dashed', label='RMS')

        # rmax = rms.max()
        # unit = 10 ** np.floor(np.log10(rmax))
        # xmax = unit * (1 + np.floor(rmax / unit))
        # ax.set_xlim(-0.1 * xmax, xmax)
        ax.set_xlim(-0.1, 1.5)

        ax.set_ylim(5, 60)
        ax.grid(color='lightgray')
        ax.tick_params('both', direction='in')

        # rms = np.sqrt((W[:, j, 0] ** 2).mean(axis=0))
        title = f'{rmses_W[0]:.4f}, {rmses_W[1]:.4f}'
        ax.set_title(title)

        if j == 0:
            ax.legend()

    plt.tight_layout()
    path = 'plots/ml-accel/training-errors.png'
    plt.savefig(path, dpi=400, bbox_inches='tight')

def plot_training_samples(kind: Literal['inputs', 'outputs']) -> None:
    """
    Plot input or output profiles from the dataset.

    Parameters
    ----------
    kind
        Whether to plot `'inputs'` or `'outputs'`.

    """

    with open('data/ml-accel/models/hyperparameters.json') as f:
        n_bins = [1, 2, 5][json.load(f)['n_bin_idx']]

    arrays = parse_integrations(cached=True)
    args = (n_bins, 'te', arrays, 1000, False)
    (C, M, Y, W), (idx_tr, _), (_, M_trans) = prepare_data(*args)

    if kind == 'inputs':
        xmaxes = [3] * n_bins
        colors = ['royalblue'] * n_bins
        data = M_trans(M[idx_tr])
        data_hat = None

    else:
        xmaxes = [0.08] + [0.02] * (n_bins - 1) + [0.001]
        colors = ['royalblue'] * n_bins + ['tab:red']
        data = (W * Y)[idx_tr]
        M = M[idx_tr]

        model = torch.jit.load('data/ml-accel/models/model-best.jit')
        Y_hat, D_hat = model(*[torch.as_tensor(a)[idx_tr] for a in (C, M)])
        data_hat = torch.cat((Y_hat, D_hat[:, None]), dim=1).numpy()

    n_rows, n_cols = 4, data.shape[1]
    fig, axes = plt.subplots(n_rows, n_cols)
    fig.set_size_inches(3 * n_cols, 4.5 * n_rows)

    z = get_vertical_grids()[1] / 1000
    rng = np.random.default_rng(1111)
    rand = rng.random(data.shape[0])
    ks = np.argsort(rand)[:n_rows]

    for i, k in enumerate(ks):
        for j, (color, xmax) in enumerate(zip(colors, xmaxes)):
            axes[i, j].plot(data[k, j], z, color=color)

            if kind == 'outputs':
                axes[i, j].plot(data_hat[k, j], z, color=color, ls='dashed')

            xmin = -(1 if kind == 'inputs' else 0.1) * xmax
            axes[i, j].set_xlim(xmin, xmax)
            axes[i, j].set_xlim(-0.0005, 0.0005)
            axes[i, j].set_ylim(5, 60)

            tmin = xmin if kind == 'inputs' else 0
            n_ticks = 5 if kind == 'inputs' else 3
            axes[i, j].set_xticks(np.linspace(tmin, xmax, n_ticks))

            axes[i, j].grid(color='lightgray')
            axes[i, j].tick_params('both', direction='in')

            if kind == 'outputs':
                a = 100 * data[k, j].sum()
                b = 100 * data_hat[k, j].sum()
                axes[i, j].set_title(f'{a:.2f}% ({b:.2f}%)')
                
            if i == n_rows - 1:
                axes[i, j].set_xlabel(kind[:-1])

            if j == 0:
                axes[i, j].set_ylabel('height (km)')

    plt.tight_layout()
    path = f'plots/ml-accel/training-{kind}.png'
    plt.savefig(path, dpi=400, bbox_inches='tight')

def plot_training_series(path: str) -> None:
    """
    Make a plot of the momentum and flux time series from an integration used to
    generate training data.

    Parameters
    ----------
    path
        Path to a netCDF file with training data. Or, can pass `'all'`, in which
        case plots of all the available integration files will be made.

    """

    if path == 'all':
        for path in iter_paths('te'):
            plot_training_series(path)

        return

    with xr.open_dataset(path) as ds:
        z_centers = ds['z_centers'].values / 1000
        z_faces = ds['z_faces'].values / 1000
        days = ds['time'].values / 86400

        M = ds['M_bulk'].sum('bin').values
        S = ds['source'].sum('bin').values
        D = ds['sink'].values

        latitude = ds.attrs['latitude']
        amaxes = [x * (1 + (abs(latitude) > 25)) for x in [0.1, 4]]

    dz = np.diff(z_faces) * 1000
    dt = np.diff(days)[0] * 86400

    F = np.zeros((len(days), 4, len(z_faces)))
    dM_dt = (M[1:] - M[:-1] + D[1:] - S[1:]) / dt
    F[1:, :, 1:] = np.cumsum(-dM_dt, axis=-1) * dz
    F[:, 1], F[:, 3] = -F[:, 1], -F[:, 3]

    widths = [4.5] * 4 + [0.2]
    fig, axes = plt.subplots(2, 5, width_ratios=widths)
    fig.set_size_inches(sum(widths), 2 * 3)
    axes, caxes = axes[:, :-1], axes[:, -1]

    datas = [M, 1000 * F]
    names = ['density', 'flux']
    units = ['kg / s / m$^2$', 'mPa']
    cmaps = [LSC.from_list('custom', ['w', 'forestgreen'], 256), 'RdBu_r']

    zipped = zip(datas, amaxes, names, units, cmaps)
    for i, (data, amax, name, unit, cmap) in enumerate(zipped):
        z = [z_centers, z_faces][i]
        amin = [0, -amax][i]

        for j in range(4):
            img = axes[i, j].pcolormesh(
                days, z, data[:, j].T,
                vmin=amin, vmax=amax,
                shading='nearest',
                cmap=cmap
            )

            axes[i, j].set_xlim(0, 30)
            axes[i, j].set_ylim(5, 60)

            if i == 1:
                axes[i, j].set_xlabel('time (days)')

            if j == 0:
                axes[i, j].set_ylabel('height (km)')

        cbar = plt.colorbar(img, cax=caxes[i])
        cbar.set_label(f'momentum {name} ({unit})')
        cbar.set_ticks(np.linspace(amin, amax, 5))

    plt.tight_layout()
    name = path.split('/')[-1].split('.')[0]
    path = f'plots/ml-accel/series/{name}.png'
    plt.savefig(path, dpi=400, bbox_inches='tight')
