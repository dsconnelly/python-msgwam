import sys

import cftime
import torch

sys.path.insert(0, '.')
from msgwam import config, spectra
from msgwam.constants import EPOCH
from msgwam.utils import open_dataset

from utils import integrate_batches

N_BATCHES = 20
PACKETS_PER_BATCH = 2000
RAYS_PER_PACKET = 10

def save_training_data():
    """
    Save training inputs and outputs. The training inputs contain zonal wind
    profiles and the properties of ray volumes to be launched. The outputs are
    time series of momentum flux obtained by integrating the trajectories of
    those ray volumes holding the mean wind constant. See the more specific
    generation functions for more details.
    """
    
    X = _sample_wave_packets()
    wind = _sample_wind_profiles()
    Z = _generate_targets(X, wind)

    torch.save(wind, 'data/coarsenet/wind.pkl')
    torch.save(X, 'data/coarsenet/spectra.pkl')
    torch.save(Z, 'data/coarsenet/targets.pkl')

def _sample_wave_packets() -> torch.Tensor:
    """
    Sample wave packets containing at most `RAYS_PER_PACKET` ray volumes from
    the spectrum defined in the config file, making sure to only sample packets
    with ray volumes all having the same sign horizontal wavenumber.

    Returns
    -------
    torch.Tensor
        Tensor of shape `(N_BATCHES, 9, PACKETS_PER_BATCH, RAYS_PER_PACKET)`
        whose first dimension ranges over batches, whose second dimension ranges
        over ray volume properties, whose third dimension ranges over wave
        packets within a batch, and whose fourth dimension ranges over
        individual rays in a particular packet.

    """

    spectrum = spectra.get_spectrum()
    spectrum_pos = spectrum[:, spectrum[2] > 0]
    spectrum_neg = spectrum[:, spectrum[2] < 0]

    shape = (N_BATCHES, 9, 2, PACKETS_PER_BATCH // 2, RAYS_PER_PACKET)
    X = torch.zeros(shape)
    
    for i in range(N_BATCHES):
        for j, half in enumerate((spectrum_pos, spectrum_neg)):
            rands = torch.rand(PACKETS_PER_BATCH // 2, half.shape[1])
            idx = torch.argsort(rands, dim=1)[:, :RAYS_PER_PACKET]
            drop = torch.rand(*idx.shape) > 0.8

            data = half[:, idx]
            data[:, drop] = torch.nan
            X[i, :, j] = data

    return X.reshape(N_BATCHES, 9, PACKETS_PER_BATCH, RAYS_PER_PACKET)

def _sample_wind_profiles() -> torch.Tensor:
    """
    Sample mean wind profiles from a reference run performed with a config file
    of the same name, except for the `'-coarsenet'` suffix. Each wind profile
    will correspond to a different batch.

    Returns
    -------
    torch.Tensor
        Tensor of shape `(N_BATCHES, 2, config.n_grid - 1)` whose first
        dimension ranges over batches, whose second dimension ranges over
        components of the mean wind, and whose third dimension ranges over
        vertical grid points.

    """

    name = config.name.replace('-coarsenet', '')
    path = f'data/{name}/reference.nc'

    with open_dataset(path) as ds:
        days = cftime.date2num(ds['time'], f'days since {EPOCH}')
        ds = ds.isel(time=((10 <= days) & (days <= 20)))

        idx = torch.randperm(len(ds['time']))[:N_BATCHES]
        u = torch.as_tensor(ds['u'].values)[idx]
        v = torch.as_tensor(ds['v'].values)[idx]

    return torch.stack((u, v), dim=1)

def _generate_targets(X: torch.Tensor, wind: torch.Tensor) -> torch.Tensor:
    """
    Integrate the model with each mean wind profile and batch of wave packets,
    and then postprocess the outputs accordingly.

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

    n_z = config.n_grid - 1
    n_snapshots = config.n_t_max // config.n_skip + 1
    Z = torch.zeros((N_BATCHES, PACKETS_PER_BATCH, n_snapshots, n_z))

    for i in range(N_BATCHES):
        spectrum = X[i].reshape(9, -1)
        integrate_batches(wind[i], spectrum, RAYS_PER_PACKET, Z[i])

    return Z.mean(dim=2)
