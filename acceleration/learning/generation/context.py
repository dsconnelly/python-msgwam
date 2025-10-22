from msgwam import config

from ... import hyperparameters as hp
from ...scenarios import get_mima_scenario

from .utils import get_info

def save_training_context() -> None:
    """Save the mean states for the training integrations."""

    year, month, site, _ = get_info(hp.task_id)
    path = f'data/ml-accel/context/{year}/{site}-{month}.nc'

    with config.override(name=f'mima-{site}'):
        get_mima_scenario(year=year, month=month).to_netcdf(path)