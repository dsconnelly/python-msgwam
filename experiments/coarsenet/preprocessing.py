import cftime
import torch

from msgwam import config
from msgwam.constants import EPOCH
from msgwam.spectra import get_spectrum
from msgwam.utils import open_dataset

from utils import integrate_batch

FIXED_WIND = False

N_BATCHES = 5
PACKETS_PER_BATCH = 32
RAYS_PER_PACKET = 9

def save_training_data() -> None:
    """
    Save training inputs and outputs. The training inputs contain zonal wind
    profiles and the properties of ray volumes to be launched. The outputs are
    time series of momentum flux obtained by integrating the trajectories of
    those ray volumes holding the mean wind constant. See the more specific
    generation functions for more details.
    """

    X, Y = _sample_packets()
    wind = _sample_wind_profiles()
    Z = _integrate_packets(wind, X, RAYS_PER_PACKET)
    Z_coarse = _integrate_packets(wind, Y, 1)

    torch.save(wind, 'data/coarsenet/wind.pkl')
    torch.save(X, 'data/coarsenet/X.pkl')
    torch.save(Y, 'data/coarsenet/Y.pkl')

    torch.save(Z, 'data/coarsenet/Z.pkl')
    torch.save(Z_coarse, 'data/coarsenet/Z_coarse.pkl')
    
def _integrate_packets(
    wind: torch.Tensor,
    X: torch.Tensor,
    rays_per_packet: int
) -> torch.Tensor:
    """
    Integrate the model with each mean wind profile and batch of wave packets,
    and then postprocess the outputs accordingly. For smoothness, the output
    time step is set to the physics time step.

    Parameters
    ----------
    wind
        Tensor structured as in the output of `_sample_wind_profiles`.
    X
        Tensor structured as one of the outputs of `_sample_wave_packets`.
    rays_per_packet
        How many rays to include in each packet during integration.

    Returns
    -------
    torch.Tensor
        Tensor of shape `(N_BATCHES, PACKETS_PER_BATCH, n_snapshots, n_z)` whose
        first dimension ranges over batches, whose second dimension ranges over
        wave packets within a batch, whose third dimension ranges over time
        steps in the integration, and whose fourth dimension ranges over
        vertical grid points.

    """

    n_z = config.n_grid - 1
    n_snapshots = (config.n_t_max - 1) // config.n_skip + 1
    Z = torch.zeros((N_BATCHES, PACKETS_PER_BATCH, n_snapshots, n_z))

    for i in range(N_BATCHES):
        spectrum = X[i].reshape(9, -1)
        Z[i] = integrate_batch(wind[i], spectrum, rays_per_packet)

    return Z

def _get_spectrum() -> torch.Tensor:
    """
    Wrapper around `spectra.get_spectrum` to convert the result to a Tensor.

    Returns
    -------
    torch.Tensor
        Tensor containing the data returned by `get_spectrum`.

    """

    return torch.as_tensor(get_spectrum().to_array().values)

def _permute_together(
    X: torch.Tensor,
    Y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Permute sampled packets between batches, so that different randomizations of
    the spectrum get mixed between wind profiles. Should be called with the fine
    and coarse data together, so that corresponding packets move together.

    Parameters
    ----------
    X, Y
        Batches of fine and coarse wave packets, respectively, as returned by
        the `_sample_wave_packets` function.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Same wave packets, but with the batch and packet dimensions shuffled.

    """

    X, Y = X.transpose(0, 1), Y.transpose(0, 1)
    jdx = torch.randperm(N_BATCHES * PACKETS_PER_BATCH)
    X, Y = X.flatten(1, 2)[:, jdx], Y.flatten(1, 2)[:, jdx]

    shape = (9, N_BATCHES, PACKETS_PER_BATCH, RAYS_PER_PACKET)
    X, Y = X.reshape(shape), Y.reshape(shape[:-1])

    return X.transpose(0, 1), Y.transpose(0, 1)

def _randomize_spectrum() -> None:
    """
    Randomize the properties of the Gaussian spectrum, within reason.
    """
    
    bounds = {
        'bc_mom_flux' : [1e-3, 5e-3],
        'wvl_hor_char' : [80e3, 120e3],
        'c_center' : [0, 15],
        'c_width' : [8, 16]
    }

    for name, (lo, hi) in bounds.items():
        sample = (hi - lo) * torch.rand(1) + lo
        setattr(config, name, sample.item())
        
    config.refresh()

def _sample_packets() -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample wave packets. First, the properties of the source spectrum are
    randomized, and then the ray volumes are divided and stacked into packets.
    The coarse analogues of each packet are also computed. Then the fine and
    coarse packets are permuted so that packets from a particular randomized
    spectrum might be integrated in different batches.

    Returns
    -------
    torch.Tensor
        Tensor of fine ray volumes whose first dimension ranges over batches,
        whose second dimension ranges over ray properties, whose third dimension
        ranges over packets in a given batch, and whose fourth dimension ranges
        over ray volumes within each packet.
    torch.Tensor
        Tensor of coarse ray volumes of the same shape as the previous, except
        without a fourth dimension.

    """

    X = torch.zeros((N_BATCHES, 9, PACKETS_PER_BATCH, RAYS_PER_PACKET))
    Y = torch.zeros((N_BATCHES, 9, PACKETS_PER_BATCH))

    for i in range(N_BATCHES):
        _randomize_spectrum()
        fine = _stack(_get_spectrum())
        fine = fine.reshape(9, -1, RAYS_PER_PACKET)

        root = int(RAYS_PER_PACKET ** 0.5)
        config.n_source //= root
        config.dr_init *= root

        config.refresh()
        coarse = _get_spectrum()
        
        samples = torch.randint(fine.shape[1], size=(PACKETS_PER_BATCH,))
        for k, sample in enumerate(samples):
            X[i, :, k] = fine[:, sample]
            Y[i, :, k] = coarse[:, sample]

        config.reset()

    r, dr, *_ = X.transpose(0, 1)
    X[:, 0] = r - r.max() + config.r_launch - 0.5 * dr
    Y[:, 0] = config.r_launch - 0.5 * Y[:, 1]

    return _permute_together(X, Y)

def _sample_wind_profiles() -> torch.Tensor:
    """
    Sample mean wind profiles from a reference run. Each wind profile will
    correspond to a different batch of wave packets.

    Returns
    -------
    torch.Tensor
        Tensor of shape `(N_BATCHES, n_time, 2, config.n_grid - 1)` whose first
        dimension ranges over batches, whose second dimension ranges over time
        steps (or is a singleton if `FIXED_WIND`), whose third dimension ranges
        over components of the mean wind, and whose fourth dimension ranges over
        vertical grid points.

    """

    n_grid = config.n_grid - 1
    n_time = 1 if FIXED_WIND else config.n_t_max
    wind = torch.zeros((N_BATCHES, n_time, 2, n_grid))

    with open_dataset(config.prescribed_wind) as ds:
        days = cftime.date2num(ds['time'], f'days since {EPOCH}')
        ds = ds.isel(time=((5 <= days) & (days <= 55)))

        hi = len(ds['time']) - n_time + 1
        starts = torch.randperm(hi)[:N_BATCHES]
        u = torch.as_tensor(ds['u'].values)
        v = torch.as_tensor(ds['v'].values)

        if FIXED_WIND:
            wind[:, :, 0] = u[starts, None]
            wind[:, :, 1] = v[starts, None]

        else:
            pairs = zip(starts, starts + n_time)
            idx = torch.vstack([torch.arange(s, e) for s, e in pairs])

            wind[:, :, 0] = u[idx]
            wind[:, :, 1] = v[idx]

    return wind

def _stack(spectrum: torch.Tensor) -> torch.Tensor:
    """
    Stack several copies of a source spectrum in physical space, and adjust the
    vertical coordinates of the ray volumes acoordingly.

    Parameters
    ----------
    spectrum
        Tensor of source ray volume properties.

    Returns
    -------
    torch.Tensor
        Tensor of stacked ray volume properties.

    """

    n = int(RAYS_PER_PACKET ** 0.5)
    spectrum = torch.repeat_interleave(spectrum, n, dim=1)
    cols = torch.arange(spectrum.shape[1])

    for i in range(n):
        jdx = torch.remainder(cols, n) == i
        spectrum[0, jdx] = spectrum[0, jdx] - i * spectrum[1, jdx]

    return spectrum