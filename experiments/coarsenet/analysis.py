import matplotlib.pyplot as plt
import torch

from msgwam import config
from msgwam.mean import MeanState

from hyperparameters import task_id
from training import DATA_DIR
from utils import integrate_batches

def plot_scores() -> None:
    """Plot training and validation scores for a trained model."""

    u = torch.load(f'{DATA_DIR}/wind.pkl')[:, 0]
    Y = torch.load(f'{DATA_DIR}/squares.pkl')
    Z = torch.load(f'{DATA_DIR}/targets.pkl')

    model = torch.jit.load(f'{DATA_DIR}/model-{task_id}.jit')
    idx_tr = torch.load(f'{DATA_DIR}/idx-tr-{task_id}.pkl')
    idx_va = torch.load(f'{DATA_DIR}/idx-va-{task_id}.pkl')

    fig, axes = plt.subplots(ncols=2)
    fig.set_size_inches(6, 4.5)

    y = torch.linspace(*config.grid_bounds, config.n_grid - 1) / 1e3
    y_mid = (y[:-1] + y[1:]) / 2

    for k, idx in enumerate([idx_tr, idx_va]):
        Y_set, Z_set = Y[..., idx], Z[:, idx]
        rmse_flux, rmse_drag = _get_loss_profiles(model, u, Y_set, Z_set)

        ls = ['dashed', 'solid'][k]
        label = ['training', 'validation'][k]
        axes[0].plot(rmse_flux * 1000, y, color='k', ls=ls)
        axes[1].plot(rmse_drag * 86400, y_mid, color='k', ls=ls, label=label)

    for ax in axes:
        ax.set_ylim(y.min(), y.max())
        ax.set_ylabel('height (km)')

    axes[0].set_xlim(0, 0.03)
    axes[1].set_xlim(0, 0.04)

    axes[0].set_xlabel('flux RMSE (mPa)')
    axes[1].set_xlabel('drag RMSE (m / s / day)')
    axes[1].legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(f'plots/coarsenet/scores-{task_id}.png', dpi=400)

def _get_loss_profiles(
    model: torch.jit.ScriptModule,
    u: torch.Tensor,
    Y: torch.Tensor,
    Z: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute flux and drag RMSE profiles across several batches.

    model
        JITted model mapping inputs to spectra.
    u
        Batches of zonal wind profiles.
    Y
        Batches of coarsened ray volumes.
    Z
        Batches of momentum flux time series.
    
    Returns
    -------
    torch.Tensor
        RMSE profile for momentum flux.
    torch.Tensor
        RMSE profile for acceleration.

    """

    state = MeanState()
    rho = (state.rho[:-1] + state.rho[1:]) / 2
    dz = state.dz

    n_batches, n_packets, _, n_grid = Z.shape
    rmse_flux = torch.zeros((n_batches, n_grid))
    rmse_drag = torch.zeros((n_batches, n_grid - 1))

    for k, (u_batch, Y_batch, Z_batch) in enumerate(zip(u, Y, Z)):
        spectrum = model(u_batch, Y_batch)
        wind = torch.vstack((u_batch, torch.zeros_like(u_batch)))
        Z_hat = integrate_batches(wind, spectrum, 1, smoothing=None)

        pmf_hat = Z_hat.mean(dim=1)
        pmf_batch = Z_batch.mean(dim=1)
        rmse_flux[k] = ((pmf_hat - pmf_batch) ** 2).sum(dim=0)

        drag_hat = _get_mean_drag(Z_hat, rho, dz)
        drag_batch = _get_mean_drag(Z_batch, rho, dz)
        rmse_drag[k] = ((drag_hat - drag_batch) ** 2).sum(dim=0)

    n_profiles = n_batches * n_packets
    rmse_flux = torch.sqrt(rmse_flux.sum(dim=0) / n_profiles)
    rmse_drag = torch.sqrt(rmse_drag.sum(dim=0) / n_profiles)

    return rmse_flux, rmse_drag

def _get_mean_drag(
    Z: torch.Tensor,
    rho: torch.Tensor,
    dz: float
) -> torch.Tensor:
    """
    Calculate the time-mean gravity wave acceleration.

    Parameters
    ----------
    Z
        Time series of momentum flux.
    rho
        Density at half vertical levels.
    dz
        Difference in height between vertical levels.
    
    Returns
    -------
    torch.Tensor
        Acceleration at each half vertical level.

    """

    dpmf_dz = torch.diff(Z, dim=-1) / dz
    return (-dpmf_dz / rho).mean(dim=1)
