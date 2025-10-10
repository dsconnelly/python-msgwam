import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr

from matplotlib.colors import LinearSegmentedColormap as LSC

from msgwam.utils import get_vertical_grids

from ..hyperparameters import architectures as hp

from .training import (
    BulkLoss,
    iter_paths,
    parse_integrations,
    prepare_data
)

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
    (C, M, Y, D), idxs, _ = prepare_data(n_bins, 'te')
    Y_tr, D_tr = torch.as_tensor(Y[idxs[0]]), torch.as_tensor(D[idxs[0]])

    loss_func = BulkLoss(Y_tr, D_tr)
    scales = (loss_func._scales_Y_tr, loss_func._scales_D[None])
    print([a.shape for a in scales])
    scales = np.concatenate([a.numpy() for a in scales], axis=0)
    print(scales.shape)

    C, M = torch.as_tensor(C), torch.as_tensor(M)
    Y_hat, D_hat = torch.jit.load(model_path)(C, M)
    Y_hat, D_hat = Y_hat.numpy(), D_hat.numpy()

    fig, axes = plt.subplots(ncols=(n_bins + 1))
    fig.set_size_inches(3 * (n_bins + 1), 4.5)
    z = get_vertical_grids()[1] / 1000

    data = np.concatenate((Y, D[:, None]), axis=1)
    data_hat = np.concatenate((Y_hat, D_hat[:, None]), axis=1)

    for j, ax, in enumerate(axes):
        for i, idx in enumerate(idxs):
            label = ['training', 'test'][i]
            color = ['forestgreen', 'tab:red'][i]
            
            diff = data[idx, j] - data_hat[idx, j]
            rmse = np.sqrt((diff ** 2).mean(axis=0))
            ax.plot(rmse, z, color=color, label=label)

        ax.plot(scales[j], z, color='gray', ls='dashed', label='scale')

        rmax = scales[j].max()
        unit = 10 ** np.floor(np.log10(rmax))
        xmax = unit * (1 + np.floor(rmax / unit))
        ax.set_xlim(-0.1 * xmax, xmax)

        ax.set_ylim(5, 60)
        ax.grid(color='lightgray')
        ax.tick_params('both', direction='in')

        if j == 0:
            ax.legend()

    plt.tight_layout()
    path = 'plots/ml-accel/training-errors.png'
    plt.savefig(path, dpi=400, bbox_inches='tight')

def plot_training_samples(
    n_bins_str: str,
    kind: str
) -> None:
    """
    Plot training samples from the dataset.

    Parameters
    ----------
    n_bins_str
        How many phase speed bins to include.
    kind
        Whether to plot `'inputs'` or `'outputs'`. Alternatively, can be a path
        to a JITted model, in which case output data will be plotted along with
        neural network predictions for each sample.

    """

    C, M, Y, D = parse_integrations('va')
    data_hats = None

    if kind not in ['inputs', 'outputs']:
        model_path = kind
        kind = 'outputs'

        model = torch.jit.load(model_path)
        Y_hat, D_hat = model(torch.as_tensor(C), torch.as_tensor(M))
        data_hats = np.concatenate((Y_hat, D_hat[:, None]), axis=1)

    n_bins = int(n_bins_str)
    (_, M, Y, D), (idx_tr, _), _ = prepare_data(n_bins, 'va', (C, M, Y, D))

    if kind == 'inputs':
        datas = M
        xmaxes = [3] * n_bins
        colors = ['royalblue'] * n_bins

    elif kind == 'outputs':
        datas = np.concatenate((Y, D[:, None]), axis=1)
        xmaxes = [0.08] + [0.02] * (n_bins - 1) + [0.001]
        colors = ['royalblue'] * n_bins + ['tab:red']

    n_rows, n_cols = datas.shape[1], 4
    fig, axes = plt.subplots(n_rows, n_cols)
    fig.set_size_inches(3 * n_cols, 4.5 * n_rows)
    
    rand = np.random.rand(len(idx_tr))
    ks = idx_tr[np.argsort(rand)[:n_cols]]
    z = get_vertical_grids()[1] / 1000

    for j, k in enumerate(ks):
        for i in range(n_rows):
            axes[i, j].plot(datas[k, i], z, color=colors[i])

            if data_hats is not None:
                axes[i, j].plot(
                    data_hats[k, i], z,
                    color=colors[i],
                    ls='dashed'
                )

            xmax = xmaxes[i]
            xmin = -(1 if kind == 'inputs' else 0.1) * xmax
            axes[i, j].set_xlim(xmin, xmax)
            axes[i, j].set_ylim(5, 60)

            tmin = xmin if kind == 'inputs' else 0
            n_ticks = 5 if kind == 'inputs' else 3
            axes[i, j].set_xticks(np.linspace(tmin, xmaxes[i], n_ticks))
            
            axes[i, j].grid(color='lightgray')
            axes[i, j].tick_params('both', direction='in')

            if kind == 'outputs':
                axes[i, j].set_title(f'{100 * datas[k, i].sum():.2f}%')

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
