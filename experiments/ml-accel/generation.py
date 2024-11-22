from __future__ import annotations
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from msgwam import config
from msgwam.dispersion import get_cp_x
from msgwam.integration import integrate
from msgwam.means import MeanState
from msgwam.sources import PacketSource
from msgwam.utils import shapiro_filter

if TYPE_CHECKING:
    from msgwam.integration import _Callback
    from msgwam.propagators import TransientPropagator

MAX_DAYS = 5
N_PACKETS = 100
SPEEDUP = 36

class EnoughPackets(Exception):
    pass

class NotEnoughPackets(Exception):
    pass

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
    z_centers = np.linspace(config.z_min, config.z_max, u.shape[1]) / 1e3
    z_faces = np.linspace(config.z_min, config.z_max, Y_fine.shape[1]) / 1e3
    T = MAX_DAYS * 86400

    for i, (j, ax) in enumerate(zip(jdx, axes)):
        k, l, m, dk, dl, dm, dens = X[j]
        action = dens * dk * dl * dm

        factor = abs(k) * action * config.dr_init / T
        colors = ['forestgreen', 'royalblue']

        handles = []
        for data, color in zip([Y_fine, Y_coarse], colors):
            line, = ax.plot(data[j] / factor, z_faces, color=color)
            handles.append(line)

        tax = ax.twiny()
        tax.plot(u[j], z_centers, color='k')

        cp_x = get_cp_x(k, l, m, config.N_ref) * np.ones_like(z_centers)
        line, = tax.plot(cp_x + u[j, 0], z_centers, color='gray', ls='dashed')
        handles.append(line)

        if i == 0:
            ax.legend(handles, ['fine', 'coarse', '$c_{\\mathrm{p}}$'])

        ax.set_xlim(-1.25, 1.25)
        ax.set_ylim(z_faces.min(), z_faces.max())
        tax.set_xlim(-50, 50)

        ax.set_xlabel('normalized flux')
        tax.set_xlabel('$\\bar{u}$ (m / s)')
        ax.set_ylabel('height (km)')

        ax.grid(color='lightgray')
        ax.tick_params('both', direction='in')

    plt.tight_layout()
    plt.savefig(f'plots/{config.name}/training-samples.png', dpi=400)

def save_training_data() -> None:
    """
    
    """

    u, X = _generate_inputs()
    Y_coarse = _generate_outputs()

    root = int(SPEEDUP ** 0.5)
    kwargs = {
        'n_max' : config.n_max * SPEEDUP,
        'n_source' : config.n_source * root,
        'dr_init' : config.dr_init / root,
        'n_chromatic' : SPEEDUP,
        'n_repeat' : root
    }

    with config.override(**kwargs):
        Y_fine = _generate_outputs()

    np.save(f'data/{config.name}/u.npy', u)
    np.save(f'data/{config.name}/X.npy', X)
    np.save(f'data/{config.name}/Y-fine.npy', Y_fine)
    np.save(f'data/{config.name}/Y-coarse.npy', Y_coarse)

def _generate_inputs() -> tuple[np.ndarray, np.ndarray]:
    """
    
    """

    u = np.zeros((N_PACKETS, config.n_grid - 1))
    X = np.zeros((N_PACKETS, 7))

    mean = MeanState.from_name('prescribed')
    source = PacketSource()
    
    i = 0
    for n_step in range(config.n_steps):
        if n_step * config.dt % config.dt_launch != 0:
            continue

        mean.step(None, n_step)
        data, _ = source.launch(mean, n_step)
        n_add = min(data.shape[1], N_PACKETS - i)

        u[i:(i + n_add)] = mean.u
        X[i:(i + n_add)] = data.T[:n_add]

        i = i + n_add
        if i == N_PACKETS:
            return u, X
        
    raise NotEnoughPackets

def _generate_outputs() -> np.ndarray:
    """
    
    """

    starts = np.inf * np.ones(N_PACKETS)
    Y = np.zeros((config.n_steps, N_PACKETS, config.n_grid))
    callback = _make_callback(starts, Y)

    try:
        _ = integrate(callback)

    except EnoughPackets:
        return config.dt * Y.sum(axis=0) / (MAX_DAYS * 86400)
    
    raise NotEnoughPackets

def _make_callback(starts: np.ndarray, Y: np.ndarray) -> _Callback:
    """
    
    """

    def callback(
        mean: MeanState,
        prop: TransientPropagator,
        n_step: int
    ) -> None:
        """
        
        """

        labels, pdx = prop._get_packet_info()
        flux = prop.k * prop.action * prop._get_cg_r(mean)
        profiles = prop._project(flux[None], prop._z_padded, pdx)[0]

        profiles = profiles[labels < N_PACKETS]
        labels = labels[labels < N_PACKETS]

        new = np.isinf(starts[labels])
        starts[labels[new]] = n_step

        keep = config.dt * (n_step - starts[labels]) < MAX_DAYS * 86400
        labels, profiles = labels[keep], profiles[keep]

        if keep.sum() == 0:
            raise EnoughPackets
        
        profiles[:, 1:-1] = shapiro_filter(profiles.T).T
        Y[n_step][labels] = profiles

    return callback
