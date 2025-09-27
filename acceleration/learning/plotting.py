import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from matplotlib.colors import LinearSegmentedColormap as LSC

from msgwam import config

_SAVE_KWARGS = {
    'dpi' : 600,
    'bbox_inches' : 'tight'
}

def plot_distributions() -> None:
    """Make box-and-whisker for the training data."""

    n_rows, n_cols = 2, 5
    fig, axes = plt.subplots(n_rows, n_cols)
    fig.set_size_inches(3 * n_cols, 4.5 * n_rows)

    with xr.open_dataset(f'data/ml-accel/training/{config.name}.nc') as ds:
        names = ds['quadrant'].values.tolist()
        z = ds['z_faces'].values / 1000
        dz = np.diff(z)[0]

        mom = ds['M_bulk'].values
        flux = ds['F_bulk'].values

        cg = np.divide(
            flux, mom,
            where=(mom > 0),
            out=np.zeros_like(mom)
        )

        mom = mom / mom.sum(axis=-1)[..., None]

    mode = 'z'

    def make_boxplot(ax, data):
        if mode == 'z':
            shift = data.mean(axis=0)
            # scale = data.std(axis=0)

            a = data.copy()
            a[a == 0] = np.nan
            scale = np.nanstd(a, axis=0)

        elif mode == 'robust':
            shift = np.median(data, axis=0)
            q25 = np.quantile(data, 0.25, axis=0)
            q75 = np.quantile(data, 0.75, axis=0)
            scale = q75 - q25

        keep = scale > 0
        data = (data - shift)[:, keep] / scale[keep]

        ax.boxplot(
            data,
            sym='',
            vert=False,
            positions=z[keep],
            widths=dz,
            medianprops={
                'color' : 'tab:red',
                'zorder' : -1
            }
        )

    for j in range(4):
        for i, data in enumerate([mom[:, j], cg[:, j]]):
            make_boxplot(axes[i, j], data)

    for i, data in enumerate([mom, cg]):
        data = data.reshape(-1, config.n_grid)
        make_boxplot(axes[i, -1], data)

    ticks = np.linspace(5, 60, 12).astype(int)
    for ax in axes.flatten():
        ax.set_xlim(-2, 2)
        ax.set_ylim(5, 60)

        ax.set_xticks(np.arange(-2, 3))
        ax.set_yticks(ticks, labels=ticks)

        ax.grid(color='lightgray')

    names = names + ['aggregate']
    for j, name in enumerate(names):
        axes[0, j].set_title(name)
        ax.set_axisbelow(True)

    plt.tight_layout()
    path = f'plots/ml-accel/{config.name}-distributions.png'
    plt.savefig(path, **_SAVE_KWARGS)

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


def plot_training_samples() -> None:
    """Plot individual profiles in the training data."""

    n_rows, n_cols = 3, 4
    fig, axes = plt.subplots(n_rows, n_cols)
    fig.set_size_inches(3 * n_cols, 4.5 * n_rows)

    with xr.open_dataset(f'data/ml-accel/training/{config.name}.nc') as ds:
        z = ds['z_faces'].values / 1000
        days = ds['time'].values / 86400

        mom = ds['M_bulk'].values
        flux = ds['F_bulk'].values

        # for _ in range(2):
        #     mom = apply_smoothing(mom)
        #     flux = apply_smoothing(flux)

        cg = np.divide(
            flux, mom,
            where=(mom > 0),
            out=np.zeros_like(mom)
        )

        # for _ in range(2):
        #     mom = apply_smoothing(mom)
        #     cg = apply_smoothing(cg)


    # k = np.random.randint(mom.shape[0])
    k = 1942
    print(f'Samples at k = {k}, day = {days[k]:.3f}')

    for j in range(4):
        for i, data in enumerate([mom[k, j], cg[k, j], 1000 * flux[k, j]]):
            color = ['royalblue', 'forestgreen', 'tab:red'][i]
            axes[i, j].plot(data, z, color=color)

            xmax = [0.2, 2, 5][i]
            xmin = 0 if i > 0 else 1e-8
            axes[i, j].set_xlim(xmin, xmax)
            axes[i, j].set_ylim(5, 60)

            if i == 0:
                axes[i, j].set_xscale('log')

            axes[i, j].grid(color='lightgray')
            axes[i, j].tick_params('both', direction='in')

    plt.tight_layout()
    path = f'plots/ml-accel/{config.name}-samples.png'
    plt.savefig(path, **_SAVE_KWARGS)    

def plot_training_series() -> None:
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
        flux = ds['F_bulk'].values

        cg = np.divide(
            flux, mom,
            where=(mom > 0),
            out=np.zeros_like(mom)
        )

    amaxes = [0.2, 1, 5]
    names = ['momentum density', '$c_\\mathrm{g}$', '$F$']
    units = ['kg / s / m$^2$', 'm / s', 'mPa']
    
    zipped = zip([mom, cg, 1000 * flux], amaxes, names, units)
    for i, (data, amax, name, unit) in enumerate(zipped):
        for j in range(4):
            color = ['royalblue', 'forestgreen', 'tab:red'][i]
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
    path = f'plots/ml-accel/{config.name}-series.png'
    plt.savefig(path, **_SAVE_KWARGS)
