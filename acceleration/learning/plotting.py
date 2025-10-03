from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr

from matplotlib.colors import LinearSegmentedColormap as LSC

from msgwam import config
from msgwam.utils import get_vertical_grids

from .. import hyperparameters as hp

from .generation import get_bin_edges
from .training import get_split, load_tensors

_SAVE_KWARGS = {
    'dpi' : 600,
    'bbox_inches' : 'tight'
}

def plot_training_errors(model_path: str) -> None:
    """
    
    """
    
    n_bins = hp.architectures.n_bins
    fig, axes = plt.subplots(1, n_bins + 1)
    fig.set_size_inches(3 * (n_bins + 1), 4.5)
    z = get_vertical_grids()[1] / 1000

    *inputs, Y = load_tensors('va')
    Y_hat = torch.jit.load(model_path)(*inputs)
    idx_tr, idx_ev = get_split(Y.shape[0], 'va')
    
    Y = 100 * Y.reshape(-1, n_bins + 1, config.n_grid - 1)
    Y_hat = 100 * Y_hat.reshape(-1, n_bins + 1, config.n_grid - 1)
    rms = np.sqrt((Y ** 2).mean(dim=0))

    for i, idx in enumerate([idx_tr, idx_ev]):
        rmse = np.sqrt(((Y_hat - Y) ** 2)[idx].mean(dim=0))
        label = ['training', 'evaluation'][i]
        color = ['forestgreen', 'tab:red'][i]

        for j, ax in enumerate(axes):
            ax.plot(rmse[j], z, color=color, label=label)

    edges = get_bin_edges()
    left = edges[:-1].reshape(n_bins, -1)[:, 0]
    right = edges[1:].reshape(n_bins, -1)[:, -1]

    for j, ax in enumerate(axes):
        ax.plot(rms[j], z, color='k', ls='dotted', label='RMS')

        ax.set_xlim(0, 0.08)
        ax.set_ylim(5, 60)

        ax.grid(color='lightgray')
        ax.tick_params('both', direction='in')
        ax.set_xlabel('RMSE (% of RMS)')
        ax.set_ylabel('height (km)')

        if j < n_bins:
            ax.set_title(f'$c_\\mathrm{{p}} \\in ({left[j]}, {right[j]})$ m / s')
            
        else:
            ax.set_title('dissipation')

    axes[0].legend()
    plt.tight_layout()
    path = 'plots/ml-accel/training-errors.png'
    plt.savefig(path, **_SAVE_KWARGS)

def plot_training_samples(model_path: Optional[str]=None) -> None:
    """
    Plot individual profiles in the training data.
    
    Parameters
    ----------
    model_path
        Path to a JITted model pipeline. If provided, the actual profiles will
        be shown alongside what the loaded model predicted for that sample.
    
    """

    n_rows, n_cols = 2, 4
    fig, axes = plt.subplots(n_rows, n_cols)
    fig.set_size_inches(3 * n_cols, 4.5 * n_rows)
    z = get_vertical_grids()[1] / 1000

    *inputs, Y = load_tensors('va', 1)
    dM = Y[:, :-(config.n_grid - 1)]
    D = Y[:, -(config.n_grid - 1):]

    datas = [dM, D]
    idx, _ = get_split(dM.shape[0], 'va')
    ks = idx[np.argsort(np.random.rand(len(idx)))[:n_cols]]

    data_hats = [None, None]
    if model_path is not None:
        *inputs, Y = load_tensors('va')
        Y_hat = torch.jit.load(model_path)(*inputs)

        dM_hat = Y_hat[:, :-(config.n_grid - 1)]
        D_hat = Y_hat[:, -(config.n_grid - 1):]

        dM_hat = dM_hat.reshape(-1, hp.architectures.n_bins, config.n_grid - 1)
        data_hats = [dM_hat.sum(axis=1), D_hat]

    for j, k in enumerate(ks):
        for i, (data, data_hat) in enumerate(zip(datas, data_hats)):
            color = ['royalblue', 'tab:red'][i]
            axes[i, j].plot(data[k], z, color=color)

            if data_hat is not None:
                axes[i, j].plot(data_hat[k], z, color=color, ls='dashed')

            xmax = 5e-3
            axes[i, j].set_xlim(-xmax, xmax)
            axes[i, j].set_ylim(5, 60)

            axes[i, j].grid(color='lightgray')
            axes[i, j].tick_params('both', direction='in')

            axes[i, j].set_title(f'{100 * data[k].sum():.6f}%')

    plt.tight_layout()
    path = f'plots/ml-accel/training-samples.png'
    plt.savefig(path, **_SAVE_KWARGS)    

def plot_training_series(site: str) -> None:
    """Plot the bulk momentum and group velocity time series."""

    n_rows, n_cols = 3, 4
    widths = [4.5] * n_cols + [0.2]
    
    fig, axes = plt.subplots(n_rows, n_cols + 1, width_ratios=widths)
    fig.set_size_inches(sum(widths), 3 * n_rows)
    axes, caxes = axes[:, :-1], axes[:, -1]

    with xr.open_dataset(f'data/ml-accel/training/{site}.nc') as ds:
        z_c = ds['z_centers'].values / 1000
        z_f = ds['z_faces'].values / 1000
        days = ds['time'] / 86400

        ds = ds.sum('bin')
        F = ds['F_bulk'].values
        M = ds['M_bulk'].values
        S = ds['source'].values
        D = ds['sink'].values

    F_est = np.zeros_like(F)
    dz = np.diff(z_f) * 1000
    dM_dt = (M[1:] - S[1:] + D[1:] - M[:-1]) / hp.generation.dt_output
    F_est[:-1, ..., 1:] = np.cumsum(-dM_dt, axis=-1) * dz

    amaxes = [0.2] + [5] * 2
    datas = [M, 1000 * F_est, 1000 * (F_est - F)]
    names = ['momentum density', '$F$ (estimated)', 'error']
    units = ['kg / s / m$^2$', 'mPa', 'mPa']

    zipped = zip(datas, amaxes, names, units)
    for i, (data, amax, name, unit) in enumerate(zipped):
        for j in range(4):
            if i < 1:
                color = 'forestgreen'
                cmap = LSC.from_list('custom', ['w', color], 256)
                amin = 0

            else:
                colors = ['royalblue', 'w', 'tab:red']
                cmap = LSC.from_list('custom', colors, 256)
                amin = -amax

            z = z_f if i > 0 else z_c
            img = axes[i, j].pcolormesh(
                days, z, data[:, j].T,
                vmin=amin, vmax=amax,
                shading='nearest',
                cmap=cmap
            )

        cbar = plt.colorbar(img, cax=caxes[i])
        cbar.set_label(f'{name} ({unit})')

    names = ['$k > 0$', '$\\ell > 0$', '$k < 0$', '$\\ell < 0$']
    for j, name in enumerate(names):
        axes[0, j].set_title(name)

    plt.tight_layout()
    path = f'plots/ml-accel/series/{site}.png'
    plt.savefig(path, **_SAVE_KWARGS)
