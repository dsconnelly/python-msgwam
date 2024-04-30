import sys

import cftime
import torch

sys.path.insert(0, '.')
from msgwam import config, spectra
from msgwam.constants import EPOCH
from msgwam.integration import SBDF2Integrator
from msgwam.rays import RayCollection
from msgwam.utils import open_dataset

from utils import pad_jagged

PROPS = RayCollection.props[:-2]

def save_training_data():
    """
    Save training inputs and outputs. The training inputs contain zonal wind
    profiles and the properties of ray volumes to be launched. The outputs are
    time series of momentum flux obtained by integrating the trajectories of
    those ray volumes holding the mean wind constant. See the more specific
    generation functions for more details.
    """
    
    X = _generate_inputs()
    Z = _generate_outputs(X)

    torch.save(X, 'data/coarsenet/X.pkl')
    torch.save(Z, 'data/coarsenet/Z.pkl')

    # from architectures import CoarseNet
    # model = CoarseNet()
    # Y = model(X)

def _generate_inputs() -> torch.Tensor:
    """
    Generate input mean zonal wind and wave packet properties. This function
    reads data from a reference run was performed with a config file of the same
    name (except for the `'-coarsenet'` suffix). 

    Returns
    -------
    torch.Tensor
        Tensor with zonal wind values in the first `config.n_grid - 1` columns
        and `config.n_source` additional columns for each of the wave properties
        listed in `PROPS`.

    """

    name = config.name.replace('-coarsenet', '')
    path = f'data/{name}/reference.nc'

    with open_dataset(path) as ds:
        days = cftime.date2num(ds['time'], f'days since {EPOCH}')
        ds = ds.isel(time=((10 <= days) & (days <= 20)))
        u = torch.as_tensor(ds['u'].values)

        keep = ds['age'].values == 0
        data = [pad_jagged(ds[prop], keep, config.n_source) for prop in PROPS]

    return torch.as_tensor(torch.hstack((u, *data)))

def _generate_outputs(X: torch.Tensor) -> torch.Tensor:
    """
    Integrate the model with each set of zonal wind and spectrum properties in
    `X`, formatted as in the output of `_generate_inputs`. 

    Parameters
    ----------
    X
        Tensor with rows containing `config.n_grid - 1` columns of zonal wind
        data and `config.n_source` columns for each ray volume property.

    Returns
    -------
    torch.Tensor
        Tensor whose first dimension ranges over time steps in the input data,
        whose second dimension ranges over time steps in the integration output,
        and whose third dimension ranges over the vertical grid where momentum
        fluxes are reported.

    """

    n_wind = config.n_grid - 1
    u, data = X[:, :n_wind], X[:, n_wind:]
    data = data.reshape(-1, len(PROPS), config.n_source)

    config.prescribed_wind = torch.zeros((2, n_wind))
    config.spectrum_type = 'custom'

    n_snapshots = config.n_t_max // config.n_skip + 1
    Z = torch.zeros((u.shape[0], n_snapshots, n_wind))

    for i in range(u.shape[0]):
        print(i)
        n_new = torch.argmax((data[i, 0] == 0).int())
        spectrum = data[i, :, :n_new]
        
        config.prescribed_wind[0] = u[i]
        spectra.custom = lambda: spectrum
        config.refresh()

        ds = SBDF2Integrator().integrate().to_dataset()
        Z[i] = torch.as_tensor(ds['pmf_u'].values)

    return Z