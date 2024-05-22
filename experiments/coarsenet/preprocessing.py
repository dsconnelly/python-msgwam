import sys

import cftime
import torch

sys.path.insert(0, '.')
from msgwam import config, spectra
from msgwam.constants import EPOCH
from msgwam.dispersion import cp_x
from msgwam.utils import open_dataset

from utils import integrate_batches

N_BATCHES = 200
PACKETS_PER_BATCH = 128
RAYS_PER_PACKET = 10

def save_training_data():
    """
    Save training inputs and outputs. The training inputs contain zonal wind
    profiles and the properties of ray volumes to be launched. The outputs are
    time series of momentum flux obtained by integrating the trajectories of
    those ray volumes holding the mean wind constant. See the more specific
    generation functions for more details.
    """
    
    X = _sample_random_packets()
    wind = _sample_wind_profiles()
    Z = _generate_targets(X, wind)

    torch.save(wind, 'data/coarsenet/wind.pkl')
    torch.save(X, 'data/coarsenet/packets.pkl')
    torch.save(Z, 'data/coarsenet/targets.pkl')

def _sample_random_packets() -> torch.Tensor:
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

def _sample_square_packets() -> torch.Tensor:
    """
    Sample wave packets containing at most `RAYS_PER_PACKET` ray volumes from
    the spectrum defined in the config file. Packets will contain only rays from
    a common square bounding box.

    Returns
    -------
    torch.Tensor
        Tensor of shape `(N_BATCHES, 9, PACKETS_PER_BATCH, RAYS_PER_PACKET)`
        whose first dimension ranges over batches, whose second dimension ranges
        over ray volume properties, whose third dimension ranges over wave
        packets within a batch, and whose fourth dimension ranges over
        individual rays in a particular packet.

    """

    squares = _squarify(spectra.get_spectrum())
    X = torch.zeros((N_BATCHES, 9, PACKETS_PER_BATCH, RAYS_PER_PACKET))

    size = (N_BATCHES, PACKETS_PER_BATCH)
    sdx = torch.randint(squares.shape[0], size=size)

    for i in range(N_BATCHES):
        data = squares[sdx[i]].transpose(0, 1)
        drop = torch.rand(*data.shape[1:]) > 0.8
        data[:, drop] = torch.nan
        X[i] = data

    return X

def _squarify(spectrum: torch.Tensor) -> torch.Tensor:
    """
    Group the ray volumes in a given source spectrum into squares according to
    the square root of `RAYS_PER_PACKET`.

    Parameters
    ----------
    spectrum
        Source spectrum as returned by `spectra.get_spectrum`.

    Returns
    -------
    torch.Tensor
        Tensor of shape `(n_squares, 9, RAYS_PER_PACKET)` whose first dimension
        ranges over square bounding boxes, whose second dimension ranges over
        ray volume properties, and whose third dimension ranges over individual
        rays within a particular square.

    """

    idx = torch.argsort(cp_x(*spectrum[2:5]))
    spectrum = spectrum[:, idx]

    root = int(RAYS_PER_PACKET ** 0.5)
    cols = torch.arange(config.n_source)
    jdx = torch.floor_divide(cols, root)
    
    n_squares = jdx.max() + 1
    out = torch.zeros((n_squares, 9, RAYS_PER_PACKET))

    for j in range(n_squares):
        packet = spectrum[:, jdx == j].repeat(1, root)
        offsets = torch.arange(root).repeat_interleave(root)
        packet[0] = packet[0] - packet[1, 0] * offsets

        out[j] = packet

    return out

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
        grid points. Values are the mean fluxes over the integration in mPa.

    """

    n_z = config.n_grid - 1
    n_snapshots = config.n_t_max // config.n_skip + 1
    Z = torch.zeros((N_BATCHES, PACKETS_PER_BATCH, n_snapshots, n_z))

    for i in range(N_BATCHES):
        spectrum = X[i].reshape(9, -1)
        integrate_batches(wind[i], spectrum, RAYS_PER_PACKET, Z[i])
        print(f'finished integrating batch {i + 1}')

    return Z