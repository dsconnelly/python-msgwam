import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from matplotlib.colors import LinearSegmentedColormap as LSC

from msgwam import config

_SAVE_KWARGS = {
    'dpi' : 400,
    'bbox_inches' : 'tight'
}

def plot_conservation() -> None:
    """Make plots checking that the conservation bound is satisfied."""

    fig, axes = plt.subplots(2, 2)
    fig.set_size_inches(2 * 4.5, 2 * 3)

    with xr.open_dataset(f'data/{config.name}/training/training.nc') as ds:
        for i, ax in enumerate(axes.flatten()):
            mom = ds['M_bulk'].isel(quadrant=i)
            cg = ds['cg_bulk'].isel(quadrant=i)

            dz = xr.ones_like(ds['z_faces'])
            dz[1:-1] = np.diff(dz['z_faces'])[0]
            dz[0] = dz[-1] = 0.5 * dz[1]

            loss = (mom * cg).isel(z_faces=-1).values * config.dt
            loss = np.minimum(loss, (mom * dz).isel(z_faces=-1).values)
            gain = ds['source'].isel(quadrant=i).values

            true = (dz * mom).sum('z_faces').values
            reckoned = true[0] * np.ones_like(true)
            reckoned[1:] += np.cumsum(gain - loss)[:-1]

            days = ds['time'] / 86400
            curve = (reckoned - true) / true

            ax.plot(days, 100 * curve, color='k', label='truth')
            ax.set_title(ds['quadrant'].isel(quadrant=i).item())

            ax.set_xlabel('day')
            ax.set_ylabel('error (%)')

            ax.set_xlim(0, days.max())
            ax.set_ylim(-2, 15)

            ax.tick_params('both', direction='in')
            ax.grid(color='lightgray')

    plt.tight_layout()
    plt.savefig(f'plots/{config.name}/conservation.png', **_SAVE_KWARGS)

def plot_training_data() -> None:
    """Plot the bulk momentum and group velocity time series."""

    n_rows, n_cols = 3, 4
    widths = [4.5] * n_cols + [0.2]
    
    fig, axes = plt.subplots(n_rows, n_cols + 1, width_ratios=widths)
    fig.set_size_inches(sum(widths), 3 * n_rows)
    axes, caxes = axes[:, :-1], axes[:, -1]

    with xr.open_dataset(f'data/{config.name}/training/training.nc') as ds:
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
    plt.savefig(f'plots/{config.name}/training.png', **_SAVE_KWARGS)
