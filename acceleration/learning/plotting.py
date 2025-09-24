import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr

from matplotlib.colors import LinearSegmentedColormap as LSC

from msgwam import config

from .architectures import SupervolumeNet

_SAVE_KWARGS = {
    'dpi' : 400,
    'bbox_inches' : 'tight'
}

def plot_conservation() -> None:
    """Make plots checking that the conservation bound is satisfied."""

    fig, ax = plt.subplots()
    fig.set_size_inches(4.5, 3)
    colors = ['tab:red', 'royalblue', 'forestgreen', 'darkviolet']

    with xr.open_dataset(f'data/ml-accel/training/{config.name}.nc') as ds:
        for i, color in enumerate(colors):
            mom = ds['M_bulk'].isel(quadrant=i)
            cg = ds['cg_bulk'].isel(quadrant=i)
            days = ds['time'] / 86400

            dz = xr.ones_like(ds['z_faces'])
            dz[1:-1] = np.diff(dz['z_faces'])[0]
            dz[0] = dz[-1] = 0.5 * dz[1]

            true = (dz * mom).sum('z_faces').values
            loss = (mom * cg).isel(z_faces=-1).values * config.dt
            loss = np.minimum(loss, (mom * dz).isel(z_faces=-1).values)
            gain = ds['source'].isel(quadrant=i).values

            bound = true[0] * np.ones_like(true)
            bound[1:] = (true + gain - loss)[:-1]
            error = (bound - true) / bound

            for _ in range(25):
                a = (3 * error[0] + error[1]) / 4
                b = (error[-2] + 3 * error[-1]) / 4

                error[1:-1] = (error[:-2] + 2 * error[1:-1] + error[2:]) / 4
                error[0] = a
                error[-1] = b

            label = ds['quadrant'].isel(quadrant=i).item().replace('l', '\\ell')
            ax.plot(days, 100 * error, color=color, lw=1, label=f'${label}$')

        ax.set_xlabel('day')
        ax.set_ylabel('gap (%)')
        ax.legend()

        ax.set_xlim(0, days.max())
        ax.set_ylim(0, 0.6)

        ax.tick_params('both', direction='in')
        ax.grid(color='lightgray')

    plt.tight_layout()
    path = f'plots/ml-accel/{config.name}-conservation.png'
    plt.savefig(path, **_SAVE_KWARGS)

def plot_network_fluxes() -> None:
    """Plot the network's predictions for a given MiMA scenario."""

    model = SupervolumeNet()
    path = 'data/ml-accel/models/state-0.pkl'
    state = torch.load(path, weights_only=True)
    model.eval().load_state_dict(state['model'])

    n_rows, n_cols = 2, 4
    widths = [4.5] * n_cols + [0.2]

    fig, axes = plt.subplots(n_rows, n_cols + 1, width_ratios=widths)
    fig.set_size_inches(sum(widths), 3 * n_rows)
    axes, caxes = axes[:, :-1], axes[:, -1]

    with xr.open_dataset(f'data/ml-accel/training/{config.name}.nc') as ds:
        z = ds['z_faces'].values / 1000
        days = ds['time'] / 86400

        for j in range(4):
            mom = ds['M_bulk'].isel(quadrant=j).values
            cg = ds['cg_bulk'].isel(quadrant=j).values
            source = ds['source'].isel(quadrant=j).values[:, None]

            sign = 1 if j < 2 else -1
            name = 'u' if j % 2 == 0 else 'v'
            wind = sign * ds[name].values

            flux = mom * cg
            axes[0, j].pcolormesh(
                days, z, sign * 1000 * flux.T,
                shading='nearest',
                vmin=-10, vmax=10,
                cmap='RdBu_r'
            )

            func = lambda a: torch.as_tensor(a[:-1])
            inputs = map(func, [mom, cg, source, wind])

            with torch.no_grad():
                mom_hat, cg_hat = model(*inputs)
                flux[1:] = (mom_hat * cg_hat).numpy()

            axes[1, j].pcolormesh(
                days, z, sign * 1000 * flux.T,
                shading='nearest',
                vmin=-10, vmax=10,
                cmap='RdBu_r'
            )

    plt.tight_layout()
    path = f'plots/ml-accel/{config.name}-network.png'
    plt.savefig(path, **_SAVE_KWARGS)

def plot_training_data() -> None:
    """Plot the bulk momentum and group velocity time series."""

    n_rows, n_cols = 3, 4
    widths = [4.5] * n_cols + [0.2]
    
    fig, axes = plt.subplots(n_rows, n_cols + 1, width_ratios=widths)
    fig.set_size_inches(sum(widths), 3 * n_rows)
    axes, caxes = axes[:, :-1], axes[:, -1]

    with xr.open_dataset(f'data/ml-accel/training/{config.name}.nc') as ds:
        z = ds['z_faces'].values / 1000
        days = ds['time'] / 86400

        mom = ds['M_bulk'].values
        cg = ds['cg_bulk'].values
        flux = 1000 * mom * cg

    amaxes = [0.25, 2, 5]
    names = ['momentum density', '$c_\\mathrm{g}$', '$F$']
    units = ['kg / s / m$^2$', 'm / s', 'mPa']
    
    zipped = zip([mom, cg, flux], amaxes, names, units)
    for i, (data, amax, name, unit) in enumerate(zipped):
        for j in range(4):

            options = ['tab:red', 'royalblue']
            color = 'purple' if i == 1 else options[j // 2]
            cmap = LSC.from_list('custom', ['w', color], 256)

            img = axes[i, j].pcolormesh(
                days, z, data[:, j].T,
                vmin=0, vmax=amax,
                shading='nearest',
                cmap=cmap
            )

        cbar = plt.colorbar(img, cax=caxes[i])
        cbar.set_label(f'{name} ({unit})')

    names = ['$k > 0$', '$\\ell > 0$', '$k < 0$', '$\\ell < 0$']
    for j, name in enumerate(names):
        axes[0, j].set_title(name)

    plt.tight_layout()
    path = f'plots/ml-accel/{config.name}-training.png'
    plt.savefig(path, **_SAVE_KWARGS)
