from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr

from matplotlib.colors import LinearSegmentedColormap as LSC

from msgwam import config
from msgwam.utils import get_vertical_grids

from ..hyperparameters import generation as hp

from .training import get_split, load_tensors

_SAVE_KWARGS = {
    'dpi' : 600,
    'bbox_inches' : 'tight'
}

def plot_training_samples(model_path: Optional[str]=None) -> None:
    """
    Plot individual profiles in the training data.
    
    Parameters
    ----------
    model_path
        Path to a JITted model pipeline. If provided, the actual profiles will
        be shown alongside what the loaded model predicted for that sample.
    
    """

    n_rows, n_cols = 3, 4
    fig, axes = plt.subplots(n_rows, n_cols)
    fig.set_size_inches(3 * n_cols, 4.5 * n_rows)

    z = get_vertical_grids()[0] / 1000
    *inputs, targets = load_tensors('va')
    
    M = targets[:, :config.n_grid]
    cg = targets[:, config.n_grid:]
    datas = [M, cg, 1000 * M * cg]

    _, idx = get_split(M.shape[0], 'TR')
    rand = np.random.rand(len(idx))
    ks = idx[np.argsort(rand)[:4]]

    if model_path is not None:
        model = torch.jit.load(model_path)
        M_hat, cg_hat = model(*inputs)
    
        data_hats = [M_hat, cg_hat, 1000 * M_hat * cg_hat]

    for j, k in enumerate(ks):
        for i, data in enumerate(datas):
            color = ['royalblue', 'forestgreen', 'tab:red'][i]
            axes[i, j].plot(data[k], z, color=color)

            if model_path is not None:
                axes[i, j].plot(data_hats[i][k], z, color=color, ls='dashed')

            xmax = 1.1 * data[k].max()
            axes[i, j].set_xlim(-0.1 * xmax, xmax)
            axes[i, j].set_ylim(5, 60)

            axes[i, j].grid(color='lightgray')
            axes[i, j].tick_params('both', direction='in')

    plt.tight_layout()
    path = f'plots/ml-accel/training-samples.png'
    plt.savefig(path, **_SAVE_KWARGS)    

def plot_training_series() -> None:
    """Plot the bulk momentum and group velocity time series."""

    n_rows, n_cols = 3, 4
    widths = [4.5] * n_cols + [0.2]
    
    fig, axes = plt.subplots(n_rows, n_cols + 1, width_ratios=widths)
    fig.set_size_inches(sum(widths), 3 * n_rows)
    axes, caxes = axes[:, :-1], axes[:, -1]

    with xr.open_dataset(f'data/ml-accel/training/{config.name}.nc') as ds:
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
    dM_dt = (M[1:] - M[:-1] - S[:-1] + D[:-1]) / hp.dt_coarse
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
    path = f'plots/ml-accel/series/{config.name}.png'
    plt.savefig(path, **_SAVE_KWARGS)
