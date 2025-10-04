from msgwam import config

from ... import hyperparameters as hp
from ...scenarios import get_mima_scenario
from ...shared.distributed import get_workload

from .utils import get_site_and_lat

def save_training_context() -> None:
    """Save the mean states for the training integrations."""

    for n in range(*get_workload(12)):
        site, _ = get_site_and_lat(n)
        
        with config.override(name=f'mima-{site}'):
            path = f'data/ml-accel/context/{site}.nc'
            get_mima_scenario(one_month=False).to_netcdf(path)