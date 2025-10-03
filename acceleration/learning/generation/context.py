from msgwam import config

from ... import hyperparameters as hp
from ...scenarios import get_mima_scenario

from .utils import get_site_and_lat

def save_training_context() -> None:
    """Save the mean states for the training integrations."""

    site, _ = get_site_and_lat(hp.task_id)
    with config.override(name=f'mima-{site}'):
        path = f'data/ml-accel/context/{site}.nc'
        get_mima_scenario(one_month=True).to_netcdf(path)