import sys

import cftime
import torch

sys.path.insert(0, '.')
from msgwam import config, spectra
from msgwam.constants import EPOCH
from msgwam.dispersion import cp_x
from msgwam.sources import CoarseSource
from msgwam.utils import open_dataset

from utils import integrate_batches

N_BATCHES = 200
PACKETS_PER_BATCH = 128

def save_training_data():
    """
    Save training inputs and outputs. The training inputs contain zonal wind
    profiles and the properties of ray volumes to be launched. The outputs are
    time series of momentum flux obtained by integrating the trajectories of
    those ray volumes holding the mean wind constant. See the more specific
    generation functions for more details.
    """
    
    X = _sample_packets()
    wind = _sample_wind_profiles()
    Z = _generate_targets(X, wind)

    torch.save(wind, 'data/coarsenet/wind.pkl')
    torch.save(X, 'data/coarsenet/packets.pkl')
    torch.save(Z, 'data/coarsenet/targets.pkl')

def _sample_packets() -> torch.Tensor:
    """
    Sample wave packets. Packets are sampled by first randomizing the spectrum,
    then stacking several copies on top of each other and grouping nearby rays
    according to the configuration state.

    Returns
    -------
    torch.Tensor
        Tensor whose first dimension ranges over batches, whose second dimension
        ranges over ray properties, whose third dimension ranges over packets in
        a given batch, and whose fourth dimension ranges over ray volumes within
        each packet.

    """

    shape = (N_BATCHES, 9, PACKETS_PER_BATCH, config.rays_per_packet)
    X = torch.zeros(shape)

    for i in range(N_BATCHES):
        spectrum = _get_random_spectrum()
        spectrum = _stack(spectrum, config.coarse_height)
        spectrum = spectrum.reshape(9, -1, config.rays_per_packet)

        samples = torch.randint(spectrum.shape[1], size=(PACKETS_PER_BATCH,))
        for k, sample in enumerate(samples):
            X[i, :, k] = spectrum[:, sample]

    return _permute_packets(X)

def _get_random_spectrum() -> None:
    """
    Randomizes the properties of the Gaussian spectrum, within reasonable
    parameters. Then returns the source ray volumes, sorted by phase speed.
    """

    bounds = {
        'bc_mom_flux' : [1e-3, 5e-3],
        'wvl_hor_char' : [20e3, 200e3],
        'c_center' : [0, 15],
        'c_width' : [8, 16]
    }

    for name, (lo, hi) in bounds.items():
        sample = (hi - lo) * torch.rand(1) + lo
        setattr(config, name, sample.item())

    config.refresh()
    spectrum = spectra.get_spectrum()
    jdx = torch.argsort(cp_x(*spectrum[2:5]))

    return spectrum[:, jdx]

def _stack(spectrum: torch.Tensor, n: int) -> torch.Tensor:
    """
    Stack several copies of a source spectrum in physical space, and adjust the
    vertical coordinates of the ray volumes accordingly.

    Parameters
    ----------
    spectrum
        Tensor of source ray volume properties.
    n
        Number of copies of the spectrum to stack.

    Returns
    -------
    torch.Tensor
        Tensor of stacked ray volume properties.

    """

    spectrum = torch.repeat_interleave(spectrum, n, dim=1)
    cols = torch.arange(spectrum.shape[1])

    for i in range(n):
        jdx = torch.remainder(cols, n) == i
        spectrum[0, jdx] = spectrum[0, jdx] - i * spectrum[1, jdx]

    return spectrum        

def _permute_packets(X: torch.Tensor) -> torch.Tensor:
    """
    Permute sampled packets between batches, so that different randomizations of
    the spectrum get mixed between wind profiles.

    Parameters
    ----------
    X
        Batches of sampled wave packets, of the form produced by one of the
        sampling functions in this module.

    Returns
    -------
    torch.Tensor
        Same wave packets, but with the batch and packet dimensions shuffled.

    """

    X = X.transpose(0, 1)
    n_packets = N_BATCHES * PACKETS_PER_BATCH
    X = X.flatten(1, 2)[:, torch.randperm(n_packets)]
    X = X.reshape(9, N_BATCHES, PACKETS_PER_BATCH, config.rays_per_packet)

    return X.transpose(0, 1)

def _sample_wind_profiles() -> torch.Tensor:
    """
    Sample mean wind profiles from a reference run performed with a config file
    of the same name, except for the `'-training'` suffix. Each wind profile
    will correspond to a different batch.

    Returns
    -------
    torch.Tensor
        Tensor of shape `(N_BATCHES, 2, config.n_grid - 1)` whose first
        dimension ranges over batches, whose second dimension ranges over
        components of the mean wind, and whose third dimension ranges over
        vertical grid points.

    """

    name = config.name.replace('-training', '')
    path = f'data/{name}/mean-state.nc'

    with open_dataset(path) as ds:
        days = cftime.date2num(ds['time'], f'days since {EPOCH}')
        ds = ds.isel(time=((5 <= days) & (days <= 25)))

        idx = torch.randperm(len(ds['time']))[:N_BATCHES]
        u = torch.as_tensor(ds['u'].values)[idx]
        v = torch.as_tensor(ds['v'].values)[idx]

    return torch.stack((u, v), dim=1)

def _generate_targets(X: torch.Tensor, wind: torch.Tensor) -> torch.Tensor:
    """
    Integrate the model with each mean wind profile and batch of wave packets,
    and then postprocess the outputs accordingly. For smoothness, the output
    time step is set to the physics time step.

    Parameters
    ----------
    X
        Tensor structured as in the output of `_sample_wave_packets`.
    wind
        Tensor structured as in the outout of `_sample_wind_profiles`.

    Returns
    -------
    torch.Tensor
        Tensor of shape `(N_BATCHES, PACKETS_PER_BATCH, n_z)` whose first
        dimension ranges over batches, whose second dimension ranges over wave
        packets within a batch, and whose third dimension ranges over vertical
        grid points. Values are the mean fluxes over the integration.

    """

    config.dt_output = config.dt
    config.refresh()
    
    n_z = config.n_grid - 1
    n_snapshots = config.n_t_max // config.n_skip + 1
    Z = torch.zeros((N_BATCHES, PACKETS_PER_BATCH, n_snapshots, n_z))    

    for i in range(N_BATCHES):
        spectrum = X[i].reshape(9, -1)
        Z[i] = integrate_batches(wind[i], spectrum, config.rays_per_packet, 4)        
        print(f'finished integrating batch {i + 1}')

    return Z